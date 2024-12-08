import torch
import time
import math
import datetime
import os
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer, AutoConfig, get_constant_schedule_with_warmup
from sklearn.metrics import classification_report


seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed_val)

class Generator(nn.Module):
    def __init__(self, noise_size=100, output_size=512, hidden_sizes=None, dropout_rate=0.1):
        super(Generator, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [512]
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.LeakyReLU(0.2, inplace=True),
                           nn.Dropout(dropout_rate)])

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep


class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=None, num_labels=2, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [512]
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.LeakyReLU(0.2, inplace=True),
                           nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers)  # per il flatten
        self.logit = nn.Linear(hidden_sizes[-1],
                               num_labels + 1)  # +1 for the probability of this sample being fake/real.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs


class Ganbert:
    def __init__(self, label_list, model="bert-base-cased", max_seq_length=64, batch_size=64,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 num_hidden_layers_g=1, num_hidden_layers_d=1, noise_size=100, out_dropout_rate=0.2,
                 apply_balance=True, learning_rate_discriminator=5e-5, learning_rate_generator=5e-5,
                 epsilon=1e-8, apply_scheduler=False, warmup_proportion=0.1):
        # 初始化标签
        self.label_list = label_list
        self.label_map = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

        # 调试输出
        print("标签列表:", self.label_list)
        print("标签映射:", self.label_map)

        # 加载 BERT 模型和 Tokenizer
        self.transformer = AutoModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = torch.device(device)

        # 初始化其他参数
        self.num_hidden_layers_g = num_hidden_layers_g
        self.num_hidden_layers_d = num_hidden_layers_d
        self.noise_size = noise_size
        self.out_dropout_rate = out_dropout_rate
        self.apply_balance = apply_balance

        self.learning_rate_discriminator = learning_rate_discriminator
        self.learning_rate_generator = learning_rate_generator
        self.epsilon = epsilon
        self.apply_scheduler = apply_scheduler
        self.warmup_proportion = warmup_proportion

        config = AutoConfig.from_pretrained(model)
        hidden_size = int(config.hidden_size)
        hidden_levels_g = [hidden_size for _ in range(0, num_hidden_layers_g)]
        hidden_levels_d = [hidden_size for _ in range(0, num_hidden_layers_d)]
        self.generator = Generator(noise_size=noise_size, output_size=hidden_size, hidden_sizes=hidden_levels_g,
                                   dropout_rate=out_dropout_rate)
        self.discriminator = Discriminator(input_size=hidden_size, hidden_sizes=hidden_levels_d,
                                           num_labels=len(label_list),
                                           dropout_rate=out_dropout_rate)

        # Put everything in the GPU if available
        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
            self.transformer.cuda()

    def generate_data_loader(self, input_examples, label_masks, label_map, do_shuffle=False, balance_label_examples=False):
        """
        Generate a Dataloader given the input examples, eventually masked if they are to be considered NOT labeled.
        """
        examples = []

        # Count the percentage of labeled examples
        num_labeled_examples = 0
        for label_mask in label_masks:
            if label_mask:
                num_labeled_examples += 1
        label_mask_rate = num_labeled_examples / len(input_examples)

        # if required it applies the balance
        for index, ex in enumerate(input_examples):
            if label_mask_rate == 1 or not balance_label_examples:
                examples.append((ex, label_masks[index]))
            else:
                # IT SIMULATE A LABELED EXAMPLE
                if label_masks[index]:
                    balance = int(1 / label_mask_rate)
                    balance = int(math.log(balance, 2))
                    if balance < 1:
                        balance = 1
                    for b in range(0, int(balance)):
                        examples.append((ex, label_masks[index]))
                else:
                    examples.append((ex, label_masks[index]))

        # Generate input examples to the Transformer
        input_ids = []
        input_mask_array = []
        label_mask_array = []
        label_id_array = []

        # Tokenization
        for (text, label_mask) in examples:
            encoded_sent = self.tokenizer.encode(text[0] + f" [EMOTION] {text[1].split('_')[0]}", add_special_tokens=True, max_length=self.max_seq_length,
                                            padding="max_length", truncation=True)
            input_ids.append(encoded_sent)
            label_id_array.append(label_map[text[1]])
            label_mask_array.append(label_mask)

        # Attention to token (to ignore padded input wordpieces)
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            input_mask_array.append(att_mask)
        # Convertion to Tensor
        input_ids = torch.tensor(input_ids)
        input_mask_array = torch.tensor(input_mask_array)
        label_id_array = torch.tensor(label_id_array, dtype=torch.long)
        label_mask_array = torch.tensor(label_mask_array)

        # Building the TensorDataset
        dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)

        if do_shuffle:
            sampler = RandomSampler
        else:
            sampler = SequentialSampler

        # Building the DataLoader
        return DataLoader(
            dataset,  # The training samples.
            sampler=sampler(dataset),
            batch_size=self.batch_size)  # Trains with this batch size.

    @staticmethod
    def format_time(elapsed):
        """
        Takes a time in seconds and returns a string hh:mm:ss
        """
        # Round to the nearest second.
        elapsed_rounded = int(round(elapsed))
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    @staticmethod
    def plot_and_store_confusion_matrix(y_true: list,
                                        y_pred: list,
                                        file_name: str,
                                        normalize=True,
                                        cmap=plt.cm.Blues,
                                        show=False):
        """
        This function prints and plots the confusion matrix, and saves it to a file
        :param y_true: The true classes
        :param y_pred: The predicted classes
        :param file_name: The file name to store the image of the confusion matrix
        :param normalize: normalize numbers (counts to relative counts)
        :param cmap: Layout
        :param show: Display the matrix. If false, only store it
        :return: Nothing
        """
        np.set_printoptions(precision=2)
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = sorted(label_list)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=[20, 27])
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(file_name)
        if show:
            plt.show()

    def train(self, labeled_data, unlabeled_data, test_data, label_list, outfile_name, tgt_label, num_train_epochs=3):
        # If there's a GPU available...
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            self.device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        # If not...
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        label_list = [f"{emotion}_{intensity}" for emotion in ['joy', 'sadness', 'fear', 'anger', 'surprise'] for intensity in range(4)]
        label_map = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for i, label in enumerate(label_list)}

        # Load the train dataset
        train_examples = labeled_data
        # The labeled (train) dataset is assigned with a mask set to True
        train_label_masks = np.ones(len(labeled_data), dtype=bool)
        # If unlabel examples are available
        if unlabeled_data:
            train_examples = train_examples + unlabeled_data
            # The unlabeled (train) dataset is assigned with a mask set to False
            tmp_masks = np.zeros(len(unlabeled_data), dtype=bool)
            train_label_masks = np.concatenate([train_label_masks, tmp_masks])
        train_dataloader = self.generate_data_loader(train_examples, train_label_masks, label_map, do_shuffle=True,
                                                     balance_label_examples=self.apply_balance)

        # Load the test dataset
        test_label_masks = np.ones(len(test_data), dtype=bool)
        test_dataloader = self.generate_data_loader(test_data, test_label_masks, label_map, do_shuffle=False,
                                                    balance_label_examples=False)
        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # models parameters
        transformer_vars = [i for i in self.transformer.parameters()]
        d_vars = transformer_vars + [v for v in self.discriminator.parameters()]
        g_vars = [v for v in self.generator.parameters()]

        # optimizer
        dis_optimizer = torch.optim.AdamW(d_vars, lr=self.learning_rate_discriminator)
        gen_optimizer = torch.optim.AdamW(g_vars, lr=self.learning_rate_generator)

        # scheduler
        if self.apply_scheduler:
            num_train_examples = len(train_examples)
            num_train_steps = int(num_train_examples / self.batch_size * num_train_epochs)
            num_warmup_steps = int(num_train_steps * self.warmup_proportion)

            scheduler_d = get_constant_schedule_with_warmup(dis_optimizer,
                                                            num_warmup_steps=num_warmup_steps)
            scheduler_g = get_constant_schedule_with_warmup(gen_optimizer,
                                                            num_warmup_steps=num_warmup_steps)
        with open(outfile_name, "w", encoding="utf-8") as outfile:
            outfile.write("#Info:\n")
            outfile.write("Classifier: {0}\n".format('GAN-BERT Classifier'))
            outfile.write("Label: {0}\n".format(tgt_label))
            outfile.write("Text label: {0}\n".format('text'))
            outfile.write("\n#Counts:\n")
            outfile.write("Number of training unlabeled_data_records: {0}\n".format(len(unlabeled_data)))
            outfile.write("Number of training labeled_data_records: {0}\n".format(len(labeled_data)))
            outfile.write("Number of classified data_records: {0}\n".format(len(test_data)))
            outfile.write("Number of unique classes in data_records: {0}\n".format(len(label_list) - 1))
            outfile.write("Number of unique classes found: {0}\n".format(len(label_list) - 1))
            outfile.write("\n#Classification report:\n")


        acc_list, g_loss_list, d_loss_list = [], [], []
        for epoch_i in range(0, num_train_epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_train_epochs))
            print('Start Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            tr_g_loss = 0
            tr_d_loss = 0

            # Put the model into training mode.
            self.transformer.train()
            self.generator.train()
            self.discriminator.train()

            # For each batch of training data...
            for step, batch in enumerate(tqdm(train_dataloader)):
                # Unpack this training batch from the dataloader.
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                b_label_mask = batch[3].to(self.device)

                real_batch_size = b_input_ids.shape[0]

                # Encode real data in the Transformer
                model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
                hidden_states = model_outputs[-1]

                # Generate fake data that should have the same distribution of the ones encoded by the transformer.
                # First noisy input are used in input to the Generator
                noise = torch.zeros(real_batch_size, self.noise_size, device=self.device).uniform_(0, 1)
                # Gnerate Fake data
                gen_rep = self.generator(noise)

                # Generate the output of the Discriminator for real and fake data.
                # First, put together the output of the tranformer and the generator
                disciminator_input = torch.cat([hidden_states, gen_rep], dim=0)
                # Then, select the output of the disciminator
                features, logits, probs = self.discriminator(disciminator_input)

                # Finally, separate the discriminator's output for the real and fake data
                features_list = torch.split(features, real_batch_size)
                D_real_features = features_list[0]
                D_fake_features = features_list[1]

                logits_list = torch.split(logits, real_batch_size)
                D_real_logits = logits_list[0]
                D_fake_logits = logits_list[1]

                probs_list = torch.split(probs, real_batch_size)
                D_real_probs = probs_list[0]
                D_fake_probs = probs_list[1]

                # Generator's LOSS estimation
                g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:, -1] + self.epsilon))
                g_feat_reg = torch.mean(
                    torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
                g_loss = g_loss_d + g_feat_reg

                # Disciminator's LOSS estimation
                logits = D_real_logits[:, 0:-1]
                log_probs = F.log_softmax(logits, dim=-1)
                # The discriminator provides an output for labeled and unlabeled real data
                # so the loss evaluated for unlabeled data is ignored (masked)
                label2one_hot = torch.nn.functional.one_hot(b_labels, len(label_list))
                per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
                per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(self.device))
                labeled_example_count = per_example_loss.type(torch.float32).numel()

                # It may be the case that a batch does not contain labeled examples,
                # so the "supervised loss" in this case is not evaluated
                if labeled_example_count == 0:
                    D_L_Supervised = 0
                else:
                    D_L_Supervised = torch.div(torch.sum(per_example_loss.to(self.device)), labeled_example_count)

                D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + self.epsilon))
                D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + self.epsilon))
                d_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U

                # Avoid gradient accumulation
                gen_optimizer.zero_grad()
                dis_optimizer.zero_grad()

                # Calculate weigth updates
                # retain_graph=True is required since the underlying graph will be deleted after backward
                g_loss.backward(retain_graph=True)
                d_loss.backward()

                # Apply modifications
                gen_optimizer.step()
                dis_optimizer.step()

                # Save the losses to print them later
                tr_g_loss += g_loss.item()
                tr_d_loss += d_loss.item()

                # Update the learning rate with the scheduler
                if self.apply_scheduler:
                    scheduler_d.step()
                    scheduler_g.step()

            # Calculate the average loss over all the batches.
            avg_train_loss_g = tr_g_loss / len(train_dataloader)
            avg_train_loss_d = tr_d_loss / len(train_dataloader)
            g_loss_list.append(avg_train_loss_g)
            d_loss_list.append(avg_train_loss_d)

            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss generetor: {0:.3f}".format(avg_train_loss_g))
            print("  Average training loss discriminator: {0:.3f}".format(avg_train_loss_d))
            print("  Training epcoh took: {:}".format(training_time))

            # After the completion of each training epoch, measure the performance on the test set.
            print("")
            print("Running Test...")
            classification_time = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
            self.transformer.eval()
            self.discriminator.eval()
            self.generator.eval()

            # Tracking variables
            total_test_accuracy = 0

            total_test_loss = 0
            nb_test_steps = 0

            all_preds = []
            all_labels_ids = []

            # loss
            nll_loss = torch.nn.CrossEntropyLoss()

            # Evaluate data for one epoch
            for batch in test_dataloader:
                # Unpack this training batch from our dataloader.
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Tell pytorch not to bother with constructing to compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
                    hidden_states = model_outputs[-1]
                    _, logits, probs = self.discriminator(hidden_states)
                    filtered_logits = logits[:, 0:-1]
                    # Accumulate the test loss.
                    total_test_loss += nll_loss(filtered_logits, b_labels)

                # Accumulate the predictions and the input labels
                _, preds = torch.max(filtered_logits, 1)
                all_preds += preds.detach().cpu()
                all_labels_ids += b_labels.detach().cpu()

            # Report the final accuracy for this validation run.
            all_preds = torch.stack(all_preds).numpy()
            all_labels_ids = torch.stack(all_labels_ids).numpy()
            predicted_classes = [id2label[pred_id] for pred_id in all_preds]
            expected_classes = [id2label[true_id] for true_id in all_labels_ids]
            test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
            acc_list.append(test_accuracy)
            print(classification_report(expected_classes, predicted_classes, digits=3))

            # Calculate the average loss over all the batches.
            avg_test_loss = total_test_loss / len(test_dataloader)
            avg_test_loss = avg_test_loss.item()

            # Measure how long the validation run took.
            classification_time = self.format_time(time.time() - classification_time)
            print("  Test Loss: {0:.3f}".format(avg_test_loss))
            print("  Test took: {:}".format(classification_time))

            with open(outfile_name, "a", encoding="utf-8") as outfile:
                outfile.write('\n - Epoch {:} / {:}\n'.format(epoch_i + 1, num_train_epochs))
                report = classification_report(expected_classes, predicted_classes, digits=3, output_dict=True)
                for label, metrics in report.items():
                    outfile.write(f"{label}: {metrics}\n")
                outfile.write(' - {0}\n'.format(
                    sklearn.metrics.classification_report(expected_classes, predicted_classes, digits=3)))
                # Print the entire confusion matrix, not truncated
                np.set_printoptions(threshold=np.inf, linewidth=200)
                outfile.write("\n - #Confusion matrix:\n{0}\n".format(
                    sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)))

                outfile.write("\n - #Performance:\n")
                outfile.write("Seconds used for training: {0}\n".format(training_time))
                outfile.write("Seconds used for classification: {0}\n".format(classification_time))

            # Also store confusion matrix as image
            imagefile_name = os.path.join('results', "{0}.jpg".format(epoch_i + 1))
            self.plot_and_store_confusion_matrix(expected_classes, predicted_classes, imagefile_name)


        with open("./results/epoch-acc.csv", "a", encoding="utf-8") as outfile:
            values = [str(value) for value in acc_list]
            outfile.write("\n".join(values))

        with open("./results/epoch-d_loss.csv", "a", encoding="utf-8") as outfile:
            values = [str(value) for value in d_loss_list]
            outfile.write("\n".join(values))

        with open("./results/epoch-g_loss.csv", "a", encoding="utf-8") as outfile:
            values = [str(value) for value in g_loss_list]
            outfile.write("\n".join(values))


        # Accuracy plot
        plt.clf()
        plt.figure(figsize=(12, 8))
        x = np.arange(1, num_train_epochs+1)
        plt.title('Accuracy')
        plt.plot(x, acc_list)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig('results/acc.png')
        plt.clf()

        # Loss plot
        plt.figure(figsize=(12, 8))
        plt.title('Average training loss')
        plt.plot(x, d_loss_list, '-',label="Discriminator")
        plt.plot(x, g_loss_list, '-',label="Generetor", color='r')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('results/loss.png')
        plt.clf()

        

