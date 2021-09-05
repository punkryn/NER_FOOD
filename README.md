# HanBert-NER

- HanBert 이용한 한국어 Named Entity Recognition Task
- 🤗`Huggingface Tranformers`🤗 라이브러리를 이용하여 구현

## Dependencies

- torch==1.4.0
- transformers==2.7.0
- seqeval==0.0.12

## Dataset

- **Naver NLP Challenge 2018**의 NER Dataset 사용 ([Github link](https://github.com/naver/nlp-challenge))
- 해당 데이터셋에 Train dataset만 존재하기에, Test dataset은 Train dataset에서 split하였습니다. ([Data link](https://github.com/aisolab/nlp_implementation/tree/master/Bidirectional_LSTM-CRF_Models_for_Sequence_Tagging/data))
  - Train (81,000) / Test (9,000)

## Details

기본적인 사용법은 [HanBert-Transformers](https://github.com/monologg/HanBert-Transformers)를 참고

### Prerequisite

- Tokenizer의 경우 현재 Ubuntu에서만 사용 가능
- HanBert Model 다운로드 (Pretrained weight + Tokenizer) 및 압축 해제
  - [HanBert-54kN-torch](https://drive.google.com/open?id=1LUyrnhuNC3e8oD2QMJv8tIDrXrxzmdu4)
  - [HanBert-54kN-IP-torch](https://drive.google.com/open?id=1wjROsuDKoJQx4Pu0nqSefVDs3echKSXP)

### Usage

```bash
$ python3 main.py --model_type hanbert \
                  --model_name_or_path HanBert-54kN-torch\
                  --do_train \
                  --do_eval

$ python3 main.py --model_type hanbert \
                  --model_name_or_path HanBert-54kN-IP-torch\
                  --do_train \
                  --do_eval
```

## Prediction

```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```

## Results

Hyperparameter는 main.py에 있는 것을 그대로 사용하였습니다

|                   | Slot F1 (%) |
| ----------------- | ----------- |
| HanBert-54kN      | **87.31**   |
| HanBert-54kN-IP   | 86.57       |
| KoBERT            | 86.11       |
| DistilKoBERT      | 84.13       |
| Bert-Multilingual | 84.20       |
| BiLSTM-CRF        | 74.57       |

## References

- [HanBert](https://github.com/tbai2019/HanBert-54k-N)
- [Naver NER result on KoBERT](https://github.com/monologg/KoBERT-NER)
- [Naver NLP Challenge](https://github.com/naver/nlp-challenge)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [NLP Implementation by aisolab](https://github.com/aisolab/nlp_implementation)
- [BERT NER by eagle705](https://github.com/eagle705/pytorch-bert-crf-ner)
