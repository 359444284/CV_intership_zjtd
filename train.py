import numpy as np
import torch


def train_epoch(
        model,
        optimizer,
        scheduler,
        device,
        loss_fn,
        train_examples,
        train_dataloader,
        val_examples=0,
        val_dataloader=None,
        epochs=10
):

    # Tracking best validation accuracy
    best_accuracy = 0

    # Start training loop
    print("Start training...\n")
    print("-" * 60)

    for epoch_i in range(epochs):
        total_loss = []
        correct_predictions = 0
        model = model.train()

        for step, batch in enumerate(train_dataloader):
            if step%500 == 499:
                print(np.mean(total_loss))

            image, label = batch
            image = image.to(device)
            label = label.to(device)


            output = model(image)

            _, preds = torch.max(output, dim=1)
            correct_predictions += torch.sum(preds == label).cpu()

            loss = loss_fn(output, label)
            total_loss.append(loss.item())


            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        avg_train_loss = np.mean(total_loss)
        train_acc = correct_predictions.double() / train_examples
        # evaluation

        if val_dataloader is not None:
            val_loss, val_acc = evaluate(model, val_examples, val_dataloader, device, loss_fn)

            # Track the best accuracy
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(model.state_dict(), 'best.bin')
                # Print performance over the entire training data
                # time_elapsed = time.time() - t0_epoch
            print([epoch_i + 1, avg_train_loss, train_acc,val_loss, val_acc])

        print("\n")
        # print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")
        print(best_accuracy)


def evaluate(model, val_examples, val_dataloader, device, loss_fn):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    correct_predictions = 0
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

            _, preds = torch.max(output, dim=1)
            correct_predictions += torch.sum(preds == label).cpu()




            # Get the predictions
    val_loss = np.mean(val_loss)

    return val_loss, correct_predictions / val_examples
