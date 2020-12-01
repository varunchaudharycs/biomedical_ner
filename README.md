# biomedical_ner
Named Entity Extraction - ADE, Reason, and Drug mention extraction - from clinical notes. Dataset = N2C2 2018- Track 2

### Pre-requisites
* Cleanup
    * Incorrect annotations 104095.ann- T17	Drug 4305 4314;4314 4315	Olanzapin e --- not in next line in txt file
    * Random characters in ann file like "" (102027, line 410)
* Data
    * track2-training_data_2
    * track2-training_data_3
    * gold-standard-test-data
* Libraries
    * preprocessing packages= ntlk, pandas, nltk.download('punkt')
    * training packages(included in notebook)
    * testing packages(included in notebook)

### Project structure

* code
    * preprocess= script to process raw data and prepare training and testing data. Involves nltk tokenization, tag filter overlap, sentence ID construction and entity statistics.
    * train= training notebook for BERT-LR model(BERT baseline)
    * test= testing notebook for BERT-LR model
    * train-crf= training notebook for BERT-CRF model(ours)
    * train-crf= testing notebook for BERT-CRF model
* data
    * track2-training_data_2= raw training data
    * gold-standard-test-data= raw test data
    * track2-training_data_3= raw validation data
    * train= processed training data
    * val= processed validation data
    * test= processed testing data
* output
    * v4= results for BERT-LR(ours- old, BERT baseline)
    * v5= results for BERT-CRF(ours)
    
### Steps

* run preprocess.py to generate data folders
* run train-crf.ipynb notebook to train BERT-CRF model
* run test-crf.ipynb notebook to test model and generate results

Note- TAG_OVERLAP_FILES(more than one entity in a span) = {"101136", "100187", "129286", "106621", "108889", "110384", "122093", "110342", "111458", "113705", "178143", "113840", "110863", "102557", "110037", "119144", "111882", "100883", "114965", "117745", "109191", "113524", "113824", "116966", "105050", "123475", "110499", "192798", "100590", "101427", "103722", "100677", "103315", "112628", "115021", "107322", "134445", "100229", "114452", "105778", "103142", "115267", "105360", "111298", "109527", "116901", "185800", "114923", "114680", "110521", "103293", "107255", "103926", "141586", "105106", "109176", "119573", "115667", "157609", "107202", "108539", "107515", "115138", "106015", "109397", "121861", "181643", "104446", "110559", "126085", "112226", "121631", "106334", "123807", "106915", "116204", "105585", "189637", "115169", "119906", "111160", "103761", "103430", "106993", "107128", "105537", "146944", "113222", "106945", "115433", "105579", "109698", "104929", "142444", "112329", "109724"}
