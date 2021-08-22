import numpy as np
import torch


def train_epoch(
        model,
        optimizer,
        scheduler,
        device,
        loss_fn,
        train_dataloader,
        val_dataloader=None,
        epochs=10
):

    # Tracking best validation accuracy
    best_accuracy = 0

    # Start training loop
    print("Start training...\n")
    print("-" * 60)

    for epoch_i in range(epochs):
        total_loss = 0
        model = model.train()

        for step, batch in enumerate(train_dataloader):

            image, label = batch
            image = image.to(device)
            label = label.to(device)


            output = model(image)
            loss = loss_fn(output, label)
            total_loss += loss.item()

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)

        # evaluation

        if val_dataloader is not None:
            val_loss, top1, top5 = evaluate(model, val_dataloader, device, loss_fn)

            # Track the best accuracy
            if top1 > best_accuracy:
                best_accuracy = top1
                # torch.save(model.state_dict(), 'best_model_state.bin')
                # Print performance over the entire training data
                # time_elapsed = time.time() - t0_epoch
            print([epoch_i + 1, avg_train_loss,val_loss, top1])

        print("\n")
        # print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")
        print(best_accuracy)


def evaluate(model, val_dataloader, device, loss_fn):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    tp_1, tp_5, counter = 0, 0, 0
    val_loss = []
    # For each batch in our validation set...
    with torch.no_grad():
        for batch in val_dataloader:

            image, label = batch
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = loss_fn(output, label)
            val_loss.append(loss.item())

            _, pred = torch.topk(output, 5, dim=1)

            correct = pred.eq(label.view(-1, 1).expand_as(pred)).cpu().numpy()
            counter += 1
            tp_1 += correct[:, 0].sum()
            tp_5 += correct.sum()



            # Get the predictions
    val_loss = np.mean(val_loss)

    return val_loss, tp_1 / counter, tp_5 / counter
