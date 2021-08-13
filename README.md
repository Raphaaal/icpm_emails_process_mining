# ICMP 2021 - Emails Process Mining

Open code for ICPM 2021 submission proposing a reproducible approach for mining business activities from emails for process analytics.

## Description

This repository contains notebooks for three dependent tasks:

1. Emails extraction from MBOX to CSV (see 0_apache_camel_email_dataset_extraction.ipynb)
2. Emails CSV preprocessing (see 1_dataset_preprocessing.ipynb)
3. Experimentations (see experimentations.ipynb, to be run in Google Colab®)

## Dependencies

Input and preprocessed data is already included in the /data/ folder. Should one need to fully reproduce these steps, one should first:
- download the raw Apache Camel MBOX files over the period 2017-04-14 10:42:39 UTC to 2017-04-19 13:27:37 UTC and store it in the /data/mailbox/ folder (available at http://mail-archives.apache.org/mod_mbox/camel-dev/201704.mbox/browser) 
- download the corresponding email labels and store them in the /data/labels/ folder (available at https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6366754/)
- download and decompress the Sense2Vec 2019 model in a /helper/ folder (available at: https://github.com/explosion/sense2vec)
- download and install SpaCy with its language version "en_core_web_sm" (available at: https://spacy.io/usage)
