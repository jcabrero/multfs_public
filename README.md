# A Methodology for Large-Scale Identification of Related Accounts in Underground Forums

This code presents the work developed for a large-scale analysis of Underground Forums. The scripts above allow the extraction of the data from a dataset and making use of that data, it is able to extract relationships between users.

## Getting started
This instructions would allow to execute the analysis on the dataset CrimeBB by Cambridge Cybercrime Centre.
### Prerequisites
The project is compiled and used under Python 3.7 version. It is not granted the functioning under other versions of the programming language. With the following command, all dependencies in requirements.txt should be installed:

```
conda create --name multfs --file requirements.txt
```
In the case that, the installation does not work that way, we can also perform the following command to install each dependency separately:
```
while read requirement; do conda install --yes $requirement; done < requirements.txt
```

In case the database is PostgreSQL as in this case, the file ```db_credentials_example.py``` must be updated to include the credentials and renamed to ```db_credentials.py```


### MULTFS extraction
The execution of the methodology requires several steps before being executed. In this project we provide the code used for the specific use case. In the case of using different features, the function ```generate_args_dict```  in ```__init__.py``` should be modified to add the new features and its locations.

In the specific case of using the CrimeBB dataset, we perform the following steps. The first and fastest option is to execute:
```
python3 __init__.py multfs
```

However, this option does not include the computation of the thresholds that are necessary. For such purpose, we should execute the methodology step by step.

By executing the following command, we extract all the users associated to features:

```
python3 dataset_generators.py datasets
```

We can specifically compute the similarity based on a single feature. For that we execute the following command:
```
python3 __init__.py compute <feature_name>
```

### Working with a different database
In the case that we make use of a different database, the features and extraction may vary. The following rules apply and need to be kept to work with the same code.

- All the files associated to one feature named ```<my_feature>```  are created and should be contained in a folder named ```<my_feature>_files```.
- This includes the initial data, which should be given in a CSV format where:
  - The name of the file should be ```user_to_<my_feature>.csv``` 
  - Each row corresponds to a user. The user name is formated according to ```username[forum_number]```.
  - We can put an arbitrary number of columns corresponding to the values of that user. Each feature value shall be formated as ```value[times_used]```.
  - An example row would be: ```foo[12], bar[3], baz[4], qux[8], quux[5], quuz[1]```

___

Authors: [@jcabrero](https://github.com/jcabrero), [@spastrana](https://github.com/spastrana)