# HanBert-NER

- [Hanbert Ner Master](https://github.com/monologg/HanBert-NER) 을 참고하였음
- HanBert 이용한 한국어 Named Entity Recognition Task + Food Tag
- Huggingface Tranformers 라이브러리를 이용하여 구현

## Dependencies

- torch==1.4.0
- transformers==2.7.0
- seqeval==0.0.12

## Dataset

- 한국해양대학교 Dataset을 사용 ([Github link](https://github.com/kmounlp/NER/tree/master/%EB%A7%90%EB%AD%89%EC%B9%98%20-%20%ED%98%95%ED%83%9C%EC%86%8C_%EA%B0%9C%EC%B2%B4%EB%AA%85))
- 해당 데이터셋에 Train dataset만 존재하기에, Test dataset은 Train dataset에서 split하였습니다. 
- 해당 데이터셋의 포맷을 Naver NLP Challenge 2018의 NER Dataset의 포맷으로 변환하였습니다.

## Model
- HanBert Model 다운로드 (Pretrained weight + Tokenizer) 및 압축 해제
  - [HanBert-54kN-torch](https://drive.google.com/open?id=1LUyrnhuNC3e8oD2QMJv8tIDrXrxzmdu4)
  - [HanBert-54kN-IP-torch](https://drive.google.com/open?id=1wjROsuDKoJQx4Pu0nqSefVDs3echKSXP)


## Results

![confusion_matrix06](https://user-images.githubusercontent.com/22855979/132113600-e8565330-f1a8-4f06-8d28-1d6ea420d217.png)
![image](https://user-images.githubusercontent.com/22855979/132115393-c7e59f6d-e09e-4d87-b6e5-a8b950d44de4.png)

F1 = 0.81 loss = 3.78


## References

- [Hanbert Ner Master](https://github.com/monologg/HanBert-NER)
