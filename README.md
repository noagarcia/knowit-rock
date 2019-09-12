## ROCK: Retrieval Over Collected Knowledge

ROCK is a model for Knowledge-Based Visual Question Answering in Videos. 
It is the first model that incorporates the use of external knowledge to answer questions about video clips.
ROCK is based on the availability of language instances representing the knowledge in a certain universe 
(e.g. a TV show), for which it constructs a knowledge base (KB). The model retrieves instances from the KB and 
fuses them with language and spatio-temporal video representations for reasoning and answer prediction.

![rock](https://github.com/noagarcia/knowit-rock/blob/master/Data/model.png?raw=true)

### Setup

1. Download the KnowIT VQA dataset from and save the csv files in `Data/`.

2. Clone the repository: 
    
    `git clone https://github.com/noagarcia/knowit-rock.git`

3. Install dependencies:
    - Python 3.6
    - numpy (`conda install -c anaconda numpy`)
    - pandas (`conda install -c anaconda pandas`)
    - sklearn (`conda install -c anaconda scikit-learn`)
    - visdom (`conda install -c conda-forge visdom`)
    - pytorch 0.4.1 (`conda install pytorch=0.4.1 cuda90 -c pytorch`)
    - pytorch-pretrained-bert 0.4.0 (`conda install -c conda-forge pytorch-pretrained-bert=0.4.0`) 
    
**Note**: Make sure to install `pytorch-pretrained-bert` instead of its newest version `pytorch-transformers`.
    
    
### ROCK Model  
    
ROCK addresses KBVQA as a multiple choice problem, 
in which each question is associated with multiple candidate answers, only one of them being correct.
The model has 3 main modules:

1. [Knowledge Base Construction](#knowledge-base): creates a knowledge base using the samples from the dataset.

2. [Knowledge Retrieval](#knowledge-retrieval): accesses the knowledge base and finds the best instance for a specific question and answers.

3. [Video Reasoning](#video-reasoning): uses the information from the video, subtitles and retrieved knoweldge to predict the correct answer.
    
    
### Knowledge Base

To create the knowledge base:

```
sh KnowledgeBase/run.sh
```


### Knowledge Retrieval

TODO.

### Video Reasoning

TODO.
    


