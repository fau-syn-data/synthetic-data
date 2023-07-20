import pandas as pd
import numpy as np
import random
import math
from typing import Callable, Union, Optional

# for graph plots
import re
import graphviz as gv
# for trackin gthe ground truth functions of the variables
import inspect

# for handling the pandas warnings
import warnings

# for adding outliers
from scipy.stats import pareto

# for imputing NA values
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer


def add_noise(n, mean = 0,sd =1):
    return np.random.normal(loc = mean, scale = sd, size= n)

# possible extensions: keep exactly the same data type e.g. int as input
def add_na(vec, n=None, share=0.05):
    result = vec.copy()
    if n is None:
        n = math.ceil(share * len(vec))
    rep = list(np.random.choice(a=range(0,len(vec)), size=n, replace=False))
    for i in rep:
        result[i] = None
    return result

# possible extensions:
# keep exactly the same data type as input e.g. int
# pass boundries e.g. only positive value
def add_outlier(vec, n=None, share=0.01):
    result = vec.copy()
    iqr = np.subtract(*np.nanpercentile(vec, [75, 25]))
    if n is None:
        n = math.ceil(share * len(vec))
    rep = list(np.random.choice(a=range(0,len(vec)), size=n, replace=False))
    for i in rep:
        result[i] = np.random.choice([
            np.nanpercentile(vec, 25) - pareto.rvs(1, loc=iqr),
            np.nanpercentile(vec, 75) + pareto.rvs(1, loc=iqr)
        ])
    return result

def target_helper(features_for_target, fun_chain, fun_args, mapping=None):
    val = None
    # print(features_for_target)
    # print(list(fun_chain.keys()))
    # print(fun_args)
    for i in range(len(features_for_target)):
        col_name = features_for_target[i].name
        # print(col_name)

        if col_name in mapping:
            feature = features_for_target[i].map(mapping[col_name]).astype("float")
        else:
            feature = features_for_target[i]
        if min(feature) == max(feature):
            scaled_feature = pd.Series([1.0] * len(feature))
        else:
            # feature = (feature - np.mean(feature)) / np.std(feature)
            scaled_feature = (feature - min(feature)) / (max(feature) - min(feature))
        if val is not None:
            vars = [val, scaled_feature, fun_args[col_name]]
        else:
            vars = [scaled_feature, fun_args[col_name]]
        val = fun_chain[list(fun_chain.keys())[i]](*vars).replace(np.Inf, np.nan)
    # print(list(val))
    return list(val)



class DataCreator:
    def __init__(self, my_df=None):
        if my_df is None:
            self.df = pd.DataFrame()
        else:
            self.df = my_df
        self.dep_dict = {}
        self.ground_truth = {}

    def generate_name(self):
        temp = "col" + str(len(self.df.columns) + 1)
        i = 2
        while temp in self.df.columns.values.tolist():
            temp = "col" + str(len(self.df.columns) + i)
        return temp

    def add_var(self, var_name:str, data_generation_function:Callable, features:list or dict=None,
                noise=True, outliers=True, nas=True):
        """Adds a variable on basis of a funcion"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            if features is not None and len(features) < data_generation_function.__code__.co_argcount:
                print("Error - Could not add the variable because too little arguments were given in your dict or list.")
            if (len(self.df.columns) == 1) and (features is None or len(features) == 0) and (self.df.columns[0] == var_name):
                #print(self.df.columns[0])
                self.df = pd.DataFrame()

            if isinstance(features, dict):
                features = {key: (
                    self.df[value].fillna(self.df[value].mean()) if is_numeric_dtype(self.df[value])
                    # else self.df[value].fillna(np.random.choice(self.df[value].mode()))
                    else self.df[value].fillna(self.df[value].mode()[0])
                ) for key, value in features.items()}
                self.df[var_name] = data_generation_function(**features)
                self.dep_dict[var_name] = [value.name for value in features.values()]
            if isinstance(features, list):
                # Replace var names with var values
                features = [(
                    self.df[var].fillna(self.df[var].mean()) if is_numeric_dtype(self.df[var])
                    # else self.df[var].fillna(np.random.choice(self.df[var].mode()))
                    else self.df[var].fillna(self.df[var].mode()[0])
                ) for var in features]
                self.df[var_name] = data_generation_function(*features)
                self.dep_dict[var_name] = [value.name for value in features]
            if features is None:
                self.df[var_name] = data_generation_function()
                self.dep_dict[var_name] = []

            if all(isinstance(e, (int, float)) for e in self.df[var_name]):
                if noise==True:
                    self.df[var_name] += add_noise(len(self.df.index),
                                                   sd=np.subtract(*np.nanpercentile(self.df[var_name], [60, 40])))
                if outliers==True:
                    self.df[var_name] = add_outlier(list(self.df[var_name]))
            if nas==True:
                self.df[var_name] = add_na(list(self.df[var_name]))

            return self

    def return_call(self):
        return self.call_str + ".create_df()\n"

    def extract_dependencies(self):

        features = [{"name": name, "level": 0, "depends_on": depends_on} for name, depends_on in self.dep_dict.items()]
        # Assign levels to the features based on dependencies
        for feature in features:
            level = 0
            for dependency in feature["depends_on"]:
                dependency_level = next((f["level"] for f in features if f["name"] == dependency), 0)
                level = max(level, dependency_level + 1)
            feature["level"] = level

        # Sort the features based on their levels
        features = sorted(features, key=lambda f: f["level"])
        return features

    def draw_graph(self):
        features = self.extract_dependencies()
        # Creates a new Digraph object
        graph = gv.Digraph(format='png', graph_attr={'rankdir': 'LR'}) # LR: left-to-right (direction of graph)
        levels = {}

        # Adds the feature to the dictionary of keys
        # Adds a node to the graph for the feature
        for feature in features:
            level = feature.get("level", 0)
            if level not in levels:
                levels[level] = []
            levels[level].append(feature)
            graph.node(feature["name"])

        # Adds an edge between each feature and the features it depends on
        for level, features in levels.items():
            for feature in features:
                for dependency in feature.get("depends_on", []):
                    graph.edge(dependency, feature["name"])

        graph.view()

    def check_inputs(self, n:int=None, var_name:str=None):
        if n is None:
            n = size=len(self.df.index)
            if n == 0:
                n = 10000
        if var_name is None:
            var_name = self.generate_name()
        return n, var_name

    def add_nominal(self, n:int=None, var_name:str=None, topic:str = "gender"):
        n, var_name = self.check_inputs(n, var_name)
        if topic=="gender":
            # print("WIP")
            # self.df[var_name] = np.random.choice(['male','female', 'diverse'], len(self.df.index), p=[0.45, 0.45, 0.1])
            return self.add_var(var_name, lambda: np.random.choice(['male','female', 'diverse'], n, p=[0.45, 0.45, 0.1]), [])
        else: return self

    def add_ordinal(self, n:int=None, var_name:str=None, topic:str = "grades"):
        n, var_name = self.check_inputs(n, var_name)
        if topic=="grades":
            # print("WIP")
            return self.add_var(var_name, lambda:
                                np.clip(
                                    np.around(
                                        np.random.normal(loc=2.5, scale=1, size=n)
                                        ), 1, 6).astype(int),
                                [])
        else: return self


    def add_interval(self, n:int=None, var_name:str=None, topic:str = "IQ"):
        n, var_name = self.check_inputs(n, var_name)
        if topic=="IQ":
            # print("WIP")
            return self.add_var(var_name, lambda:
                                np.clip(
                                    np.around(
                                        np.random.normal(loc=100, scale=15, size=n)
                                        ), 10, 250).astype(int),
                                [])
        else: return self

    def add_ratio(self, n:int=None, var_name:str=None, topic:str = "revenue"):
        n, var_name = self.check_inputs(n, var_name)
        if topic=="revenue":
            # print("WIP")
            return self.add_var(var_name, lambda:
                                np.clip(
                                  np.around(
                                        np.random.lognormal(np.log(1000000.00), 0.75, size=n), 2
                                        ), 0.00, 572754000000.00),
                                [])
        else: return self

    def gen_target(self, var_name="target", dependency_rate = 0.1, mandatory_features = [], bias = []):
        categorical_mapping = {}
        features_for_target = []
        biased_columns = []
        # print(features_for_target)
        # print(mandatory_features)
        # print(biased_columns)
        features_for_target.clear()
        biased_columns.clear()
        biased_columns.extend(bias)

        target_name = var_name
        numeric_columns = [col_name for col_name in self.df.select_dtypes(include=np.number).columns if col_name not in [target_name, "biased_" + target_name]]
        categorical_columns = [col_name for col_name in self.df.select_dtypes(include=['object']).columns if col_name not in [target_name, "biased_" + target_name]]


        number_of_dependencies = math.ceil(len(numeric_columns + categorical_columns)*dependency_rate)
        if number_of_dependencies <= 0: number_of_dependencies = 1
        min = self.df.min(axis=None, numeric_only=True).min()/2
        max = self.df.max(axis=None, numeric_only=True).max()/2
        #create numeric values for categorical columns
        for col_name in categorical_columns:
            categorical_values = self.df[col_name].unique()
            categorical_mapping[col_name] = {v: random.uniform(min, max) for v in categorical_values}

        #add mandatory features and bias to features_for_target
        features_for_target.extend(mandatory_features)
        features_for_target.extend(biased_columns)

        if len(features_for_target) < number_of_dependencies:
            features_for_target.extend(
                np.random.choice([col_name for col_name in numeric_columns + categorical_columns if col_name not in features_for_target],
                                 size=number_of_dependencies-len(features_for_target), replace=False))
        random.shuffle(features_for_target)
        #check if biased_columns is given and add one if there was none given
        if len(biased_columns) == 0:
            biased_columns.append(np.random.choice(features_for_target))

        b_fun_chain = {}
        u_fun_chain = {}
        b_fun_args = {}
        u_fun_args = {}
        # print("this is the BIASed func_chain at start: ")
        # print(b_fun_chain)
        # print("unbiased: ")
        # print(u_fun_chain)
        b_ground_truth = ""
        u_ground_truth = ""
        random_weight = np.random.uniform(0, 2.0)
        random_column = features_for_target[0]
        b_fun_chain["fun_0"] = lambda x0, rw0: x0 * rw0
        u_fun_chain["fun_0"] = lambda x0, rw0: x0 * rw0


        i = 0
        while random_column in biased_columns and i < len(features_for_target)-1:
            i += 1
            random_column = features_for_target[i]
        if i > 0:
            features_for_target[i] = features_for_target[0]
            features_for_target[0] = random_column
        b_fun_args[random_column] = random_weight
        u_fun_args[random_column] = random_weight

        b_ground_truth = "(" + b_ground_truth + str(random_weight) + " * " + random_column + ")"
        u_ground_truth = "(" + u_ground_truth + str(random_weight) + " * " + random_column + ")"

        for l in range(1, len(features_for_target)):
            var_str = ""
            random_operation = np.random.choice(["+", "-", "*", "/"])
            random_operation = "+"
            random_weight = np.random.uniform(-2.0, 2.0)
            random_column = features_for_target[l]
            # for m in range(0, l - 1):
            #   var_str += "x" + str(m) + ", " + "rw" + str(m) + ", "
            # var_str += "x" + str(l - 1) + ", " + "rw" + str(l - 1)
            # # print(var_str)
            if random_column in categorical_columns:
                b_ground_truth = "(" + b_ground_truth + " " + random_operation + " " + str(random_weight) + " * c_" + random_column + ")"
            else:
                b_ground_truth = "(" + b_ground_truth + " " + random_operation + " " + str(random_weight) + " * " + random_column + ")"
            # b_fun_chain["fun_" + str(l)] = eval("lambda " + var_str + ", x" + str(l) +  ", rw" + str(l) +
            #                                     ": x" + str(l-1) + " " +
            #                                     random_operation + " x" + str(l) + " * rw" + str(l))
            b_fun_chain["fun_" + str(l)] = eval("lambda intercept, x, rw: intercept " + random_operation + " [el * rw for el in x]")
            b_fun_args[random_column] = random_weight

            if features_for_target[l] not in biased_columns:
                n = len(u_fun_chain.keys())
                # print(n)
                var_str = ', '.join(u_fun_chain[list(u_fun_chain.keys())[-1]].__code__.co_varnames)
                # print("unbiased" + u_fun_chain[list(u_fun_chain.keys())[-1]].__code__.co_varnames[-2])
                if random_column in categorical_columns:
                    u_ground_truth = "(" + u_ground_truth + " " + random_operation + " " + str(random_weight) + " * c_" + random_column + ")"
                else:
                    u_ground_truth = "(" + u_ground_truth + " " + random_operation + " " + str(random_weight) + " * " + random_column + ")"
                # u_fun_chain["fun_" + str(n)] = eval("lambda " + var_str + ", x" + str(n) +  ", rw" + str(n) +
                #                                 ": x" + str(n-1) + " " +
                #                                 random_operation + " x" + str(n) + " * rw" + str(n))

                u_fun_chain["fun_" + str(n)] = eval("lambda intercept, x, rw: intercept " + random_operation + " [el * rw for el in x]")
                u_fun_args[random_column] = random_weight
                # print(u_fun_chain)
                # print(u_fun_args)
        self.ground_truth["biased_target"] = b_ground_truth
        self.ground_truth["unbiased_target"] = u_ground_truth

        (self.add_var(target_name, lambda *args: target_helper(args, u_fun_chain, u_fun_args, categorical_mapping), list(u_fun_args.keys()),
                      noise=False, outliers=False, nas=False)
        .add_var("biased_" + target_name, lambda *args: target_helper(args, b_fun_chain, b_fun_args, categorical_mapping), list(b_fun_args.keys()),
                      noise=False, outliers=False, nas=False))
        
        structure = "numerical"

        if structure == "categorical":
            structure_limits = []
            n_classes = 4
            bin_min = self.df[target_name].min()
            bin_max = self.df[target_name].max()
            structure_limits = [bin_min + (i+1) * (bin_max - bin_min) / n_classes for i in range(n_classes - 1)]
            # print([bin_min - abs(bin_max)] + structure_limits + [bin_max + abs(bin_min)])
            # print(pd.cut(self.df[target_name], bins= [bin_min - abs(bin_max)] + structure_limits + [bin_max + abs(bin_min)], labels= ["class_" + str(i + 1) for i in range(n_classes)] ))
            self.df[target_name] = pd.cut(self.df[target_name], bins= [bin_min - abs(bin_max)] + structure_limits + [bin_max + abs(bin_min)], labels= ["class_" + str(i + 1) for i in range(n_classes)] )
            bin_min = self.df["biased_" + target_name].min()
            bin_max = self.df["biased_" + target_name].max()
            # print(pd.cut(self.df["biased_" + target_name], bins= [bin_min - abs(bin_max)] + structure_limits + [bin_max + abs(bin_min)], labels= ["class_" + str(i + 1) for i in range(n_classes)] ))
            self.df["biased_" + target_name] = pd.cut(self.df["biased_" + target_name], bins= [bin_min - abs(bin_max)] + structure_limits + [bin_max + abs(bin_min)], labels= ["class_" + str(i + 1) for i in range(n_classes)] )
        if structure == "binary":
            bin_min = self.df[target_name].min()
            bin_max = self.df[target_name].max()
            # print(bin_min)
            # print(bin_max)
            self.df[target_name] = self.df[target_name].apply(lambda x: 0 if x <= (bin_min + (bin_max - bin_min)/2) else 1) 
            self.df["biased_" + target_name] = self.df["biased_" + target_name].apply(lambda x: 0 if x <= (bin_min + (bin_max - bin_min)/2) else 1) 
            # print((                
                # self.df[target_name].apply(lambda x: 0 if x <= (bin_min + (bin_max - bin_min)/2) else 1) 
                # self.df[target_name].where(
                #     self.df[target_name] > (bin_min + (bin_max - bin_min)/2), 1, inplace=True)
                # .where(
                #     self.df[target_name] <= (bin_min + (bin_max - bin_min)/2), 0, inplace=True)
                #.astype('int')
            # ))
                  

        return(self)


    def add_target(self, var_name:str=None, topic:str = "random"):
        """Ads a target variable based on the specified topic.
          In the future the relationships of previous variables should be specified to lead to the target."""
        n = self.df.shape[0]
        if var_name is None:
            var_name = "target"

        number_of_columns = self.df.shape[1]
        if topic=="loan":
            # print("WIP")
            return self.add_var(var_name, lambda: np.random.choice([0, 1], n, p=[0.76, 0.24]), [])
        elif topic=="random":
            # return self.add_var(var_name, lambda: self.generate_target(0.3), [])
            return self.gen_target(var_name)
        else: return self



    def gen_multi(self, n:int=None, num_cols:int=0, cat_cols:int=0, n_layers:int=None, topic="loan", biased=True, interaction="auto"):
        """Ads columns based on the number of how many categorical or numerical columns were asked for."""
        # df_trian, df_unbiased, df_real = datasetCreator(samples=10000, num_cols=6, cat_cols=3, topic="loan", biased=True, interaction="auto")
        # print("WIP")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            n = self.check_inputs(n)[0]
            if num_cols + cat_cols == 0:
                num_cols = 10
                cat_cols = 10
            if n_layers is None:
                n_layers = int(math.floor(math.log(num_cols + cat_cols) / math.log(2)))
            # print(n_layers)
            # generate the total pool of variables to be generated
            pool = ["num"]*num_cols + ["cat"]*cat_cols

            # level 1 has half of the remaining columns
            # save the information about what types of columns should be created in a dictionary
            multi_dict = {}
            # get the names of the previous dataframe to compare to later
            prev_cols = self.df.columns.values.tolist()
            for i in range(0, n_layers):
                # draw randomly from the ramaining types of variables
                # if the last level would only consist of one variable then it gets added to the previous level
                if i == n_layers - 1:
                    multi_dict["level_" + str(i)] = list(np.random.choice(a=pool, size=len(pool), replace=False))
                else:
                    multi_dict["level_" + str(i)] = list(np.random.choice(a=pool, size=math.ceil(len(pool)/2), replace=False))
                for item in multi_dict["level_" + str(i)]:
                    pool.remove(item)
                    # print("Start of createing the variable")
                    if i != 0:
                        n_rest = np.random.choice(a=[0,1,2,3,4],p=[0.25,0.30,0.25,0.15,0.05])
                        # print(n_rest)
                        # print(prev_cols)
                        # print(np.random.choice(a=prev_cols, size=2))
                        # print(np.random.choice(a=multi_dict["level_" + str(i-1)]))
                        if n_rest == 0:
                            dep = [np.random.choice(a=multi_dict["level_" + str(i-1)])]
                        else:
                            # print([np.random.choice(a=multi_dict["level_" + str(i-1)])] + [1,2,3,4])
                            # print(np.random.choice(a=prev_cols, size=n_rest))
                            dep = list(set([np.random.choice(a=multi_dict["level_" + str(i-1)])] + np.random.choice(a=prev_cols, size=n_rest).tolist()))
                        random.shuffle(dep)
                        fun_chain = {}
                        fun_args = {}
                        categorical_mapping = {}

                        min = self.df.min(axis=None, numeric_only=True).min()/2
                        max = self.df.max(axis=None, numeric_only=True).max()/2
                        # print(self.df[dep[0]].dtype)
                        if self.df[dep[0]].dtype not in (np.float64, np.int64):
                            categorical_values = self.df[dep[0]].unique().tolist()
                            categorical_mapping[dep[0]] = {v: random.uniform(min, max) for v in categorical_values}
                            # categorical_mapping[dep[0]] = {v: categorical_values.index(v) + 1 for v in categorical_values}

                        fun_args[dep[0]] = np.random.uniform(-1.0, 1.0) # np.random.normal()
                        fun_chain[dep[0]] = lambda x, rw: x * rw
                        # print(dep[0])
                        # print(fun_args[dep[0]])

                        for dep_var in dep[1:len(dep)]:
                            # print(dep_var)
                            fun_args[dep_var] = np.random.uniform(-1.0, 1.0) # np.random.normal()
                            # print(fun_args[dep_var])
                            if self.df[dep_var].dtype not in (np.float64, np.int64):
                                categorical_values = self.df[dep_var].unique().tolist()
                                # {k:i for i,k in enumerate(dictionary.keys())}
                                categorical_mapping[dep_var] = {v: random.uniform(min, max) for v in categorical_values}
                                # categorical_mapping[dep_var] = {v: categorical_values.index(v) + 1 for v in categorical_values}
                            fun_chain[dep_var] = np.random.choice([
                                lambda intercept, x, rw: intercept + x * rw,
                                #lambda intercept, x, rw: intercept - x * rw,
                                #lambda intercept, x, rw: intercept * x * rw,
                                #lambda intercept, x, rw: np.divide(intercept, x * rw, out=np.zeros_like(intercept), where=intercept!=0)
                                #np.divide(a, b, out=np.zeros_like(a), where=b!=0)
                            ])
                        
                        # print(fun_chain)
                        temp_name = self.generate_name()
                        # print("creating new variable: " + temp_name)
                        # print(categorical_mapping)
                        self.add_var(temp_name, lambda *args: target_helper(args, fun_chain, fun_args, categorical_mapping), dep,
                          noise=False, outliers=False, nas=False)
                        
                        # for categorical values transform to 0-1 encoding
                        if item =="cat":
                              structure_limits = []
                              n_classes = 4
                              bin_min = self.df[temp_name].min()
                              bin_max = self.df[temp_name].max()
                              structure_limits = [bin_min + (i+1) * (bin_max - bin_min) / n_classes for i in range(n_classes - 1)]
                              # print([bin_min - abs(bin_max)] + structure_limits + [bin_max + abs(bin_min)])
                              # print(pd.cut(self.df[temp_name], bins= [bin_min - abs(bin_max)] + structure_limits + [bin_max + abs(bin_min)], labels= ["class_" + str(i + 1) for i in range(n_classes)] ))
                              self.df[temp_name] = pd.cut(self.df[temp_name], bins= [bin_min - abs(bin_max)] + structure_limits + [bin_max + abs(bin_min)], labels= ["class_" + str(i + 1) for i in range(n_classes)] )


                    else:
                        if item == "num":
                            self.add_ratio(n)
                        if item == "cat":
                            self.add_nominal(n)
                # print("level_"+ str(i) + " num: " + str(multi_dict["level_" + str(i)].count("num")) + "\nlevel_"+ str(i) + " cat: " + str(multi_dict["level_" + str(i)].count("cat"))
                #     + "\npool num: " + str(pool.count("num")) + "\npool cat: " + str(pool.count("cat")))
                multi_dict["level_" + str(i)] = list(set(self.df.columns.values.tolist()).difference(prev_cols))
                prev_cols += multi_dict["level_" + str(i)]
                # print(multi_dict["level_" + str(i)])
                # print(pool)

            self.add_target(topic="random")
            return self

    def get_ground_truth(self):
        print(
            "The biased ground truth function is given by: " + self.ground_truth["biased_target"] + "\n" +
            "The unbiased ground truth function is given by: " + self.ground_truth["unbiased_target"] + "\n"
        )


    def create_df(self):
        return self.df.copy()