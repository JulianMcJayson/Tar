from dataset import get_dataset
from models.clip_model import ClipModel
from models.gemma_model import GemmaModel
from models.llava_model import LLaVaProjection, LLaVaBatchDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

# [batch, token, feature dimension] = [1,50,768]
# matching only batch 1 and 1 and dimension 768 to 1152(gemma3 1b)
vision_model = ClipModel()
vision_size = vision_model.get_model_size()

for p in vision_model.vision.parameters():
    p.requires_grad = False

ll_model = GemmaModel()
llm_size = ll_model.get_model_size()

for p in ll_model.llm.parameters():
    p.requires_grad = False

x_train, x_test = get_dataset(limit=10000)
print(x_train, x_test)

dataset_train = LLaVaBatchDataset(x_train, vision=vision_model, llm=ll_model)
dataset_test = LLaVaBatchDataset(x_test, vision=vision_model, llm=ll_model)

train_loader = DataLoader(dataset_train, 4, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset_test, 4, shuffle=True, num_workers=0)

projector = LLaVaProjection(vision_size, llm_size).to(vision_model.vision.device)

projector_device = next(projector.parameters()).device

projector.train()

optimizer = torch.optim.Adam(projector.parameters(), lr=2e-3)
epoch = 1

for e in range(epoch):
    total_loss = 0.0
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {e+1}")): #type: ignore
        pixel = batch["pixel_values"].to(projector_device)
        input_ids = batch["input_ids"].to(ll_model.llm.device)

        optimizer.zero_grad()

        embed = ll_model.llm.get_input_embeddings()
        Hq = embed(input_ids)

        with torch.no_grad():
            vision_out = vision_model.vision(pixel_values=pixel)
            image_feature = vision_out.last_hidden_state

        Hv : torch.Tensor = projector(image_feature)

        HvHq = torch.cat([Hv, Hq], dim=1)

        img_token = 50

        labels : torch.Tensor = input_ids.clone()

        img_mask = torch.full((labels.shape[0], img_token), -100, dtype= labels.dtype).to(labels.device)

        out_label = torch.cat((img_mask, labels), dim=1)

        outputs = ll_model.llm(input_embeds=HvHq, labels=out_label)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.items()

    avg_loss = total_loss / x_train.num_rows
    print(f"Epoch {e+1} completed. Average Loss: {avg_loss:.4f}")

torch.save(projector.state_dict(), "checkpoint.pth")