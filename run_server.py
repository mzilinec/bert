import argparse
import os

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

import modeling
import tokenization
from server_utils import create_model, softmax


class BertWrapper:

    def __init__(self, args):

        tf.logging.set_verbosity(tf.logging.DEBUG)

        if not os.path.isdir(args.model_dir):
            raise Exception("The model directory {} doesn't exist!".format(args.model_dir))

        if not os.path.isfile(args.bert_config_file):
            raise Exception("The BERT config file {} doesn't exist!".format(args.model_dir))

        if not os.path.isfile(args.vocab_file):
            raise Exception("The vocab file {} doesn't exist!".format(args.model_dir))

        self.args = args
        self.bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=args.vocab_file, do_lower_case=args.lowercase)

        self.run_config = tf.estimator.RunConfig(
            model_dir=args.model_dir,
            session_config=tf.ConfigProto(device_count={'GPU': 0}),
        )

        self.input_ids = tf.placeholder(shape=[1, args.max_seq_length], dtype=tf.int32)
        self.input_mask = tf.placeholder(shape=[1, args.max_seq_length], dtype=tf.int32)
        self.token_type_ids = tf.placeholder(shape=[1, args.max_seq_length], dtype=tf.int32)

        self.start_logits, self.end_logits = create_model(
            bert_config=self.bert_config,
            is_training=False,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            segment_ids=self.token_type_ids,
            use_one_hot_embeddings=False
        )
        self.saver = tf.train.Saver()
        self.session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.saver.restore(self.session, os.path.join(args.model_dir, args.init_checkpoint))

    def predict(self, question, context):

        max_len = args.max_seq_length
        input_mask = np.zeros([1, max_len])
        segment_ids = np.zeros([1, max_len])

        q = self.tokenizer.tokenize(question)
        c = self.tokenizer.tokenize(context)

        # Join and add control tokens
        in_seq = ["[CLS]"] + q + ["[SEP]"] + c + ["[SEP]"]

        split = len(q) + 2
        end = len(in_seq)
        delta = max_len - len(in_seq)

        # Pad the sequence
        if delta > 0:
            in_seq += ["[PAD]"] * delta
        elif delta < 0:
            in_seq = in_seq[:max_len]  # TODO: what if the question is too long?

        assert in_seq[split] == c[0]

        input_ids = [self.tokenizer.convert_tokens_to_ids(in_seq)]
        input_mask[0, 0:end] = 1
        segment_ids[0, split:end] = 1

        start_logits, end_logits = self.session.run(
            (self.start_logits, self.end_logits), {
                self.input_ids: input_ids,
                self.input_mask: input_mask,
                self.token_type_ids: segment_ids
            })

        start_probs = np.array(softmax(start_logits[0, split:end]))
        end_probs = np.array(softmax(end_logits[0, split:end]))

        start_probs = start_probs.reshape([1, end - split])
        end_probs = end_probs.reshape([1, end - split])

        possible = np.matmul(start_probs.T, end_probs)

        for i in range(possible.shape[0]):
            for j in range(possible.shape[1]):
                if i > j:
                    possible[i, j] = 0.0

        max_prob = np.max(possible)
        start, end = np.unravel_index(np.argmax(possible, axis=None), possible.shape)
        print("Max prob:", max_prob, "@", (start, end))
        print(c[start:end])
        return max_prob, " ".join(c[start:end])


app = Flask(__name__)


@app.route("/", methods=['POST'])
def answer():
    data = request.get_json(force=True)
    if 'question' not in data or 'context' not in data:
        return "Question or context not provided", 400
    question, context = data['question'], data['context']
    score, answer = bert.predict(question, context)
    return jsonify({"confidence": score, "answer": answer})


def test():
    bert.predict("How many people live in Prague?", "1500000 people live in Prague.")
    q = "Which NFL team represented the NFC at Super Bowl 50?"
    c = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."
    bert.predict(q, c)


if __name__ == "__main__":
    # ~ 800 MB of RAM required
    argp = argparse.ArgumentParser()
    argp.add_argument("--bert-config-file", type=str, required=True)
    argp.add_argument("--model-dir", type=str, required=True)
    argp.add_argument("--vocab-file", type=str, required=True)
    argp.add_argument("--init-checkpoint", type=str, required=True)

    argp.add_argument("--lowercase", type=bool, default=True, help="Set to false for cased models")
    argp.add_argument("--max-seq-length", type=int, default=320)
    argp.add_argument("--doc-stride", type=int, default=128)
    argp.add_argument("--max_query_length", type=int, default=64)

    argp.add_argument("--host", type=str, default="0.0.0.0")
    argp.add_argument("--port", type=int, default=8010)

    args = argp.parse_args()
    global bert
    bert = BertWrapper(args)
    #test()
    app.run(host=args.host, port=args.port)
