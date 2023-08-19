# NetSolP-1.0
NetSolP-1.0 predicts the solubility and usability for purification of proteins expressed in E. coli. The usability objective includes the solubility and expressibility of proteins. NetSolP-1.0 is based on protein language models (ESM12, ESM1b).

The webserver can be found at https://services.healthtech.dtu.dk/service.php?NetSolP. A standalone version of the software is available from the Downloads tab.
It contains the code for the webserver. Additionally, it also has the datasets, code for training and testing, and the trained models

## Usage

For training:
```
cd TrainAndTest/
python train.py
```
more details and requirements in the README of folder TrainAndTest.

For prediction: (First train and convert the models to ONNX OR download the pre-trained models from the webserver Downloads tab)
```
cd PredictionServer/ 
python predict.py --FASTA_PATH ./test_fasta.fasta --OUTPUT_PATH ./test_preds.csv --MODEL_TYPE ESM12 --PREDICTION_TYPE S
```
more details and requirements in the README of folder PredictionServer.

## License

The code is licensed under the [BSD 3-Clause license](https://github.com/TviNet/NetSolP-1.0/blob/main/LICENSE).

## Citation

```
@article{10.1093/bioinformatics/btab801,
    author = {Thumuluri, Vineet and Martiny, Hannah-Marie and Almagro Armenteros, Jose J and Salomon, Jesper and Nielsen, Henrik and Johansen, Alexander Rosenberg},
    title = "{NetSolP: predicting protein solubility in Escherichia coli using language models}",
    journal = {Bioinformatics},
    volume = {38},
    number = {4},
    pages = {941-946},
    year = {2021},
    month = {11},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab801},
    url = {https://doi.org/10.1093/bioinformatics/btab801},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/38/4/941/49008876/btab801.pdf},
}

```

