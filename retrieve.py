import json
import argparse
import requests

from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from task_distill import (
    ColaProcessor,
    MnliProcessor,
    MnliMismatchedProcessor,
    MrpcProcessor,
    Sst2Processor,
    StsbProcessor,
    QqpProcessor,
    QnliProcessor,
    RteProcessor,
    WnliProcessor
)

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor
}


def get_retrieved_docs(
    query, k=10, query_strategy='rake', index_name='index_128', url='https://2377-58-246-142-34.ngrok.io/search'
):
    headers = {
        'Content-type': 'application/json',
    }
    data = {"index_name": index_name, 
            "query": query, 
            "k": k, 
            "query_strategy": query_strategy}

    return requests.post(url, headers=headers, data=json.dumps(data)).json()


def main(args):
    print('- k: ', args.k)
    print('- task_name: ', args.task_name)
    print('- output_file: ', args.output_file)
    
    processor = processors[args.task_name]()
    train_examples = processor.get_train_examples(args.data_dir)

    with open(args.output_file, 'w') as wf:
        for e in tqdm(train_examples):
            try:
                if e.text_a is not None:
                    for doc in get_retrieved_docs(e.text_a, k=args.k)['texts']:
                        for s in sent_tokenize(doc.strip()):
                            wf.write(s + '\n')
                        wf.write('\n')

                if e.text_b is not None:
                    for doc in get_retrieved_docs(e.text_b, k=args.k)['texts']:
                        for s in sent_tokenize(doc.strip()):
                            wf.write(s + '\n')
                        wf.write('\n')
                wf.flush()
            except:
                print('Error!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name",
                        default='mrpc',
                        type=str,
                        required=True,
                        help="GLUE task name.")
    parser.add_argument("--data_dir",
                        default='glue_data/MRPC',
                        type=str,
                        required=True,
                        help="GLUE data directory.")
    parser.add_argument("--output_file",
                        default='mrpc_retrived.txt',
                        type=str,
                        required=True,
                        help="File to write retrived docs.")
    parser.add_argument("--k",
                        default=10,
                        type=int)

    args = parser.parse_args()
    main(args)