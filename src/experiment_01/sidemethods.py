import torch.nn.functional as F
import torch
import json
import pandas as pd
from tqdm import tqdm


def logprobs_from_prompt(prompt, tokenizer, model):
  encoded = tokenizer(prompt, return_tensors="pt")
  input_ids = encoded["input_ids"]
  output = model(input_ids=input_ids)
  shift_labels = input_ids[..., 1:].contiguous()
  shift_logits = output.logits[..., :-1, :].contiguous()
  log_probs = []
  log_probs.append((tokenizer.decode(input_ids[0].tolist()[0]), None))
  for idx, (label_id, logit) in enumerate(zip(shift_labels[0].tolist(), shift_logits[0])):
    logit = logit.type(torch.FloatTensor) # device_map="auto" fpr model initialization
    logprob = F.log_softmax(logit, dim=0)[label_id].item()
    log_probs.append((tokenizer.decode(label_id), float(logprob)))
  return log_probs

def proc(sent):
    if not sent.endswith(".") or sent.endswith("!"):  # finish with period
        sent += '.'
    if not sent[0].isupper():  # start with a capital letter
        sent = sent[0].upper() + sent[1:]
    return sent

def proc_lower(sent):
    if not sent.endswith(".") or sent.endswith("!"):  # finish with period
        sent += '.'
    if not sent[0].islower():  # start with a lowercase letter
        sent = sent[0].lower() + sent[1:]
    return sent

def prob_of_ending(token_logprobs, tokens):
    logprob_sum = 0
    for count, (lp, t) in enumerate(zip(token_logprobs[::-1], tokens[::-1])):
        if count > 0 and t.endswith('.'):
            break
        logprob_sum += lp
    return logprob_sum / count


def calculate_accuracy(fname):
    with open(fname) as f:
        logprobs = json.load(f)

    correct = 0
    for qid_label, (end1, end2) in logprobs.items():
        end1_prob = prob_of_ending(end1['token_logprobs'], end1['tokens'])
        end2_prob = prob_of_ending(end2['token_logprobs'], end2['tokens'])
        label = int(qid_label[-1])
        if (label == 0 and end1_prob > end2_prob) or (label==1 and end1_prob < end2_prob):
            correct += 1

    print(f"correct: {correct}/{len(logprobs)} = {round(correct/len(logprobs),5)}")

def calculate_accuracies(fname):
    with open(fname) as f:
        logprobs = json.load(f)

    correct = 0
    for qid_label, (end1, end2) in logprobs.items():
        end1_prob = prob_of_ending(end1['token_logprobs'], end1['tokens'])
        end2_prob = prob_of_ending(end2['token_logprobs'], end2['tokens'])
        label = int(qid_label[-1])
        if (label == 0 and end1_prob > end2_prob) or (label==1 and end1_prob < end2_prob):
            correct += 1
            print(qid_label+" correct.")
        else:
            print(qid_label+" incorrect.")

    print(f"correct: {correct}/{len(logprobs)} = {round(correct/len(logprobs),5)}")

def return_accuracy(fname):
    with open(fname) as f:
        logprobs = json.load(f)

    correct = 0
    for qid_label, (end1, end2) in logprobs.items():
        end1_prob = prob_of_ending(end1['token_logprobs'], end1['tokens'])
        end2_prob = prob_of_ending(end2['token_logprobs'], end2['tokens'])
        label = int(qid_label[-1])
        if (label == 0 and end1_prob > end2_prob) or (label==1 and end1_prob < end2_prob):
            correct += 1

    return round(correct/len(logprobs),5)



def store_accuracies(json_in, data_in, csv_out):
    with open(json_in) as f:
        logprobs = json.load(f)

    df = pd.read_csv("../data/"+data_in+".csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df[df.valid==1] 
    #print(df.head(3))

    with open("../data/"+csv_out+".csv", "w") as f_out:
      f_out.write(",".join(list(df.columns.values))+",lm_guess_correct\n")

      correct = 0
      results_qid = dict()

      
      for qid_label, (end1, end2) in logprobs.items():
          end1_prob = prob_of_ending(end1['token_logprobs'], end1['tokens'])
          end2_prob = prob_of_ending(end2['token_logprobs'], end2['tokens'])
          label = int(qid_label[-1])
          if (label == 0 and end1_prob > end2_prob) or (label==1 and end1_prob < end2_prob):
              correct += 1
              results_qid[qid_label] = 1
          else:
              results_qid[qid_label] = 0

      
      for index, row in df.iterrows():
        out_line = ""
        qid_indexed = str(row['qid'])+"_"+str(row['labels'])

        if qid_indexed in results_qid.keys():
          values = [str(v) for v in df.values.tolist()[index]]
          sentences = "\""+values[1]+"\",\""+values[2]+"\",\""+values[3]+"\","
          out_line = str(values[0])+","+sentences+",".join(values[4:])+","+str(results_qid[qid_indexed])

          #print(out_line)
          f_out.write(out_line+"\n")
