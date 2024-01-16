# Emails Process Mining

Open code accompanying the paper: [A Reproducible Approach for Mining Business Activities from Emails for Process Analytics](https://link.springer.com/chapter/10.1007/978-3-031-14135-5_6), from Raphael Azorin, Daniela Grigori, and Khalid Belhajjame at ICSOC AI-PA 2021, The 2nd International Workshop on AI-enabled Process Automation.

## Description

This repository contains notebooks for three dependent tasks:

1. Raw emails extraction from MBOX to CSV (see [0_apache_camel_email_dataset_extraction.ipynb](https://github.com/Raphaaal/icpm_emails_process_mining/blob/main/0_apache_camel_email_dataset_extraction.ipynb))
2. Emails CSV preprocessing (see [1_dataset_preprocessing.ipynb](https://github.com/Raphaaal/icpm_emails_process_mining/blob/main/1_dataset_preprocessing.ipynb))
3. Experimentations (see [experimentations.ipynb](https://github.com/Raphaaal/icpm_emails_process_mining/blob/main/experimentations.ipynb), to be run in Google Colab®)

The last task can be run independently as the preprocessed input data is already included in this repository at [/data/camel_emails_emb_s2v.csv](https://github.com/Raphaaal/icpm_emails_process_mining/blob/main/data/camel_emails_emb_s2v.csv).

## Usage

### Experience reproduction

In order to simply reproduce the experimentation results presented in the paper, one should:

1. Upload the input CSV of preprocessed emails (available at [/data/camel_emails_emb_s2v.csv](https://github.com/Raphaaal/icpm_emails_process_mining/blob/main/data/camel_emails_emb_s2v.csv)) in Google Drive®
2. Upload the experimentation notebook [experimentations.ipynb](https://github.com/Raphaaal/icpm_emails_process_mining/blob/main/experimentations.ipynb) in Google Colab®
3. Setup and run the notebook. Please note that the default parameters are those used in the article. The required setup concerns the location of input and output files on your drive.


### Preprocessing reproduction

Preprocessed input data is already included in the /data/ folder. Should one need to fully reproduce these data preparation steps, one should first:

- download the [raw Apache Camel MBOX files](http://mail-archives.apache.org/mod_mbox/camel-dev/201704.mbox/browser) over the period 2017-04-14 10:42:39 UTC to 2017-04-19 13:27:37 UTC and store it in the /data/mailbox/ folder. 
- download the [corresponding email labels](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6366754/) and store them in the /data/labels/ folder.
- download and decompress the [Sense2Vec 2019](https://github.com/explosion/sense2vec) model in the /helper/ folder.
- download and install [SpaCy](https://spacy.io/usage) with its language version "en_core_web_sm".

Then, the whole process of preprocessing the raw emails is contained in the [0_apache_camel_email_dataset_extraction.ipynb](https://github.com/Raphaaal/icpm_emails_process_mining/blob/main/0_apache_camel_email_dataset_extraction.ipynb) and [1_dataset_preprocessing.ipynb](https://github.com/Raphaaal/icpm_emails_process_mining/blob/main/1_dataset_preprocessing.ipynb) notebooks.
