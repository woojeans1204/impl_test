import os
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import yaml
import torch
import open_clip
import torchvision.datasets as torchvision_datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def denormalize(tensor, mean, std):
    mean_tensor = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std_tensor = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std_tensor + mean_tensor

def main(config_path="zeroshot.yaml"):
    config_path = "../configs/" + config_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"../results/exp_{timestamp}"
    os.makedirs(os.path.join(result_dir, "samples"), exist_ok=True)
    
    with open(os.path.join(result_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config['model']['name']
    pretrained = config['model']['pretrained']
    
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)

    dataset_name = config['data']['name']
    DatasetClass = getattr(torchvision_datasets, dataset_name)
    dataset = DatasetClass(root=config['data']['root'], train=False, download=True, transform=preprocess)
    
    dataloader = DataLoader(dataset, batch_size=config['data']['batch_size'], num_workers=config['data']['num_workers'])

    classes = dataset.classes
    
    # MNIST 클래스 이름 깔끔하게 다듬기 ("0 - zero" -> "zero")
    if dataset_name == "MNIST":
        classes = [c.split(" - ")[0] for c in classes]
        
    templates = config['prompts']
    
    target_total_samples = config.get('logging', {}).get('num_samples', 10)
    target_correct = target_total_samples // 2
    target_wrong = target_total_samples - target_correct
    
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classes:
            texts = [template.format(classname) for template in templates]
            try:
                # 일반적인 방식 시도
                texts_tokens = tokenizer(texts).to(device)
            except Exception:
                # SigLIP/T5 호환용: tokenizer.tokenizer를 직접 참조하여 인코딩
                from open_clip import get_tokenizer
                # 만약 model_name이 SigLIP 계열이라면 아래처럼 직접 처리
                tokens = tokenizer.tokenizer(texts, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
                texts_tokens = tokens['input_ids'].to(device)
            
            class_embeddings = model.encode_text(texts_tokens)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            
            zeroshot_weights.append(class_embedding)
            
        text_features = torch.stack(zeroshot_weights, dim=1).to(device)

    correct = 0
    total = 0
    saved_correct = 0
    saved_wrong = 0
    sample_logs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            logits = 100.0 * image_features @ text_features
            predictions = logits.argmax(dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            for i in range(images.size(0)):
                is_correct = (predictions[i] == labels[i]).item()
                
                if is_correct and saved_correct < target_correct:
                    status = "correct"
                    saved_correct += 1
                elif not is_correct and saved_wrong < target_wrong:
                    status = "wrong"
                    saved_wrong += 1
                else:
                    continue
                    
                true_class = classes[labels[i]]
                pred_class = classes[predictions[i]]
                
                img_name = f"{status}_{true_class}_pred_{pred_class}_{saved_correct + saved_wrong}.png"
                img_path = os.path.join(result_dir, "samples", img_name)
                
                img_denorm = denormalize(images[i].clone(), CLIP_MEAN, CLIP_STD)
                save_image(img_denorm, img_path)
                
                sample_logs.append({
                    "file_name": img_name,
                    "status": status,
                    "true_label": true_class,
                    "predicted_label": pred_class
                })

    accuracy = 100.0 * correct / total
    print(f"[{dataset_name}] Zero-shot Accuracy with Ensemble: {accuracy:.2f}%")
    
    result_data = {
        "dataset": dataset_name,
        "model": model_name,
        "pretrained": pretrained,
        "accuracy": round(accuracy, 2),
        "total_images": total,
        "samples_log": sample_logs
    }
    
    with open(os.path.join(result_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # config_path = "CIFAR10.yaml"
    # config_path = "ImageNet.yaml"
    config_path = "MNIST2.yaml"

    main(config_path)