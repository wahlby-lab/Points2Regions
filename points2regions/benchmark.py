import optuna
import time

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union, List, Any
from sklearn.metrics import adjusted_rand_score as rand_score

class BenchmarkDataset:
    """
    Class for handling benchmark datasets.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    x_col : str
        Column name for x values.
    y_col : str
        Column name for y values.
    label_col : str
        Column name for labels.
    gt_col : str
        Column name for ground truth.
    other : Any, optional
        Additional data. Defaults to None.
    """

    def __init__(self,
        data: pd.DataFrame, 
        x_col: str, 
        y_col: str, 
        label_col: str,
        gt_col: str,
        other:Any=None,
    ) -> None:
        """
        Initialize BenchmarkDataset with input data and columns.

        Convert input DataFrame to Numpy arrays.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame.
        x_col : str
            Column name for x values.
        y_col : str
            Column name for y values.
        label_col : str
            Column name for labels.
        gt_col : str
            Column name for ground truth.
        other : Any, optional
            Additional data. Defaults to None.
        """
          
        # Store inputs
        self.data = data.copy()
        self.x_col = x_col
        self.y_col = y_col
        self.label_col = label_col
        self.gt_col = gt_col

        # Convert to Numpy land
        self.x = data[x_col].to_numpy()
        self.y = data[y_col].to_numpy()
        self.label = data[label_col].to_numpy()
        self.gt = data[gt_col].to_numpy()
        self.xy = data[[x_col, y_col]].to_numpy()
        self.other = other

def create_dataset_list(
    data: pd.DataFrame,
    dataset_index_col: str, 
    x_col: str, 
    y_col: str, 
    label_col: str,
    gt_col: str
) -> List[BenchmarkDataset]:
    # Create dataset for current density factor
    datasets = []
    for _, data_by_id in data.groupby(dataset_index_col):
        datasets.append(
            BenchmarkDataset(data_by_id, x_col=x_col, y_col=y_col, label_col=label_col, gt_col=gt_col)
        )
    return datasets


class AbstractTuner:
    """
    Abstract class for tuning and benchmarking clustering algorithms.
    """

    
    def __init__(
        self
    ) -> None:
        super().__init__()
    
    def parameter_space(self, trial: optuna.Trial)->Dict[str,optuna.Trial]:
        """
        Define the parameter space for tuning.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna Trial object.

        Returns
        -------
        Dict[str, optuna.Trial]
            Dictionary of hyperparameter values.
        """
        raise NotImplementedError
    
    def cluster(self, dataset: BenchmarkDataset, **kwargs) -> np.ndarray:
        """
        Cluster the given dataset.

        Parameters
        ----------
        dataset : BenchmarkDataset
            Input dataset.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Cluster assignments.
        """
        raise NotImplementedError

    def cluster_multiple_datasets(self, datasets: List[BenchmarkDataset], **kwargs) -> List[np.ndarray]:
        """
        Cluster multiple datasets.

        Parameters
        ----------
        datasets : List[BenchmarkDataset]
            List of input datasets.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[np.ndarray]
            List of cluster assignments.
        """
        return [self.cluster(dataset, **kwargs) for dataset in datasets]
 
    def benchmark(
            self, 
            dataset: Union[BenchmarkDataset, List[BenchmarkDataset]], n_trials: int = 300, 
            optuna_optimize_kwargs: Dict[str,any]={},
            seed: int = 10,
            raise_errors:bool=False
        ) -> Tuple[Dict[str,Any], Dict[str,Any]]:
     
        """
        Perform benchmarking and tuning of the clustering algorithm.

        Parameters
        ----------
        dataset : Union[BenchmarkDataset, List[BenchmarkDataset]]
            Input dataset or list of datasets.
        n_trials : int, optional
            Number of optimization trials. Defaults to 300.
        optuna_optimize_kwargs : Dict[str, Any], optional
            Additional kwargs for Optuna optimization. Defaults to {}.
        seed : int, optional
            Random seed. Defaults to 10.
        raise_errors : bool, optional
            Whether to raise errors during tuning. Defaults to False.

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            Tuple containing best hyperparameters and benchmark results.
        """

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(
            lambda trial: self.__criterion(dataset, trial, debug=raise_errors), 
            n_trials=n_trials, 
            **optuna_optimize_kwargs
        )

        best_value, wall_clock_times = self.evaluate(dataset, study.best_params, debug=False)
        output_dict = {
            'Rand-score' : best_value,
            'Wall-clock time [s]' : wall_clock_times 
        }
        return study.best_params, output_dict
    
    def evaluate(self, dataset: Union[BenchmarkDataset, List[BenchmarkDataset]], hyperparams: Dict[str,any], debug: bool=False) -> Tuple[List[float],List[float]]:
        """
        Evaluate the clustering algorithm on the given dataset(s).

        Parameters
        ----------
        dataset : Union[BenchmarkDataset, List[BenchmarkDataset]]
            Input dataset or list of datasets.
        hyperparams : Dict[str, Any]
            Hyperparameters for the clustering algorithm.

        Returns
        -------
        Tuple[List[float], List[float]]
            Tuple containing Rand scores and wall clock times.
        """
        batch_mode = isinstance(dataset,list)
        dataset_iterator = [dataset] if not batch_mode else dataset
        rand_scores = []
        wall_clock_times = []
        for ds in dataset_iterator:
            start_time = time.time()
            if debug:
                predicted = self.cluster(ds, **hyperparams)
            else:
                try:
                    predicted = self.cluster(ds, **hyperparams)
                except:
                    print('Crashed')
                    predicted = np.zeros(len(ds.xy), dtype='int')
            end_time = time.time()
            elapsed_time = end_time - start_time

            #predicted = self.__match_with_gt(ds.gt, predicted)
            rand_scores.append(rand_score(ds.gt, predicted))
            wall_clock_times.append(elapsed_time)
        return rand_scores, wall_clock_times
    
    def __criterion(self, dataset: Union[BenchmarkDataset, List[BenchmarkDataset]], trial: optuna.trial.Trial, debug:bool=False) -> float:  
        hyperparams = self.parameter_space(trial)
        rand, _ = self.evaluate(dataset, hyperparams, debug=debug)
        return np.mean(rand)



