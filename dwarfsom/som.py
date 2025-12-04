from somoclu import Somoclu
import pandas as pd
import numpy as np

class DwarfSOM:
    def __init__(
            self,
            n_columns: int=10,
            n_rows: int=10,
            maptype: str="toroid",
            gridtype: str="hexagonal",
            neighborhood: str="gaussian",
            std_coeff: float=0.5,
            vect_distance: str="euclidean",
            initialization: str="pca",
            compactsupport: bool=False,
    ):
        self.n_columns = n_columns
        self.n_rows = n_rows
        self.somoclu = Somoclu(
            n_columns=n_columns,
            n_rows=n_rows,
            maptype=maptype,
            gridtype=gridtype,
            neighborhood=neighborhood,
            std_coeff=std_coeff,
            vect_distance=vect_distance,
            initialization=initialization,
            compactsupport=compactsupport,
        )
        self.scaler = None
        self.n_dim = None
        self.bmus_train = None
        self.is_trained = False

        self.n_labels = None
        self.bmus_label = None
        self.is_labeled = False
        self._neuron_label_data = {}

    def train(
            self,
            X: pd.DataFrame,
            scaler: callable=None,
            epochs: int=100,
    ):
        self.scaler = scaler
        X_scaled = self.scaler.fit_transform(X)
        self.somoclu.train(
            data=X_scaled,
            epochs=epochs,
        )
        self.n_dim = self.somoclu.n_dim
        self.bmus_train = self.somoclu.bmus[:, ::-1]
        self.is_trained = True


    def label_with(self, y: pd.DataFrame) -> None:
        """Assign labels to the SOM neurons based on additional data dimensions."""
        assert self.is_trained

        if y.shape[1] <= self.n_dim:
            raise ValueError(
                f"Data must have at least {self.n_dim + 1} dimensions."
            )

        self.n_labels = y.shape[1] - self.n_dim

        data_scaled = self.scaler.transform(y.iloc[:, :self.n_dim])
        activation_map = self.somoclu.get_surface_state(data_scaled)
        bmus = self.somoclu.get_bmus(activation_map)
        self.bmus_label = bmus[:, ::-1]

        for row in range(self.n_rows):
            for col in range(self.n_columns):
                bmus_in_neuron = np.where(
                    (self.bmus_label[:, 0] == row) & (self.bmus_label[:, 1] == col)
                )[0]
                self._neuron_label_data[(row, col)] = y.iloc[
                    bmus_in_neuron, self.n_dim:
                ].values.transpose()

        self.is_labeled = True
        print(f"Labeled SOM with {', '.join(y.columns[self.n_dim:].tolist())} data.")


    def count_bmus_per_neuron(self, bmus: np.ndarray | None=None) -> np.ndarray:
        """Count the number of BMUs per neuron.

        Args:
            bmus: Array of shape (n_samples, 2) containing the BMU coordinates
                for each sample. Usually obtained from `somoclu.bmus`.

        Returns:
            counts: Array of shape (n_rows, n_columns) containing the counts
                of BMUs per neuron.
        """
        assert self.is_trained
        if bmus is None:
            bmus = self.bmus_train

        counts = np.zeros(
            shape=(self.n_rows, self.n_columns),
            dtype=np.int32,
        )
        np.add.at(counts, (bmus[:, 0], bmus[:, 1]), 1)
        return counts


    def get_label_maps(self, aggfunc: callable=np.mean) -> np.ndarray:
        """Generate label maps for each label dimension.

        Args:
            aggfunc: Function to aggregate label data within each neuron.
        """
        assert self.is_labeled

        label_maps = np.empty(shape=(self.n_labels, self.n_rows, self.n_columns))
        for row in range(self.n_rows):
            for col in range(self.n_columns):
                label_data = self._neuron_label_data[(row, col)]
                if label_data.size == 0:
                    label_maps[:, row, col] = np.nan
                    continue
                else:
                    label_maps[:, row, col] = aggfunc(label_data, axis=1)

        return label_maps


    def assign_label_values(self, bmus, label_maps) -> np.ndarray:
        """Assign label values to samples based on their BMUs and label maps.

        Args:
            bmus: Array of shape (n_samples, 2) containing the BMU coordinates
                for each sample.
            label_maps: Array of shape (n_labels, n_rows, n_columns) containing
                the label maps.

        Returns:
            assigned_labels: Array of shape (n_samples, n_labels) containing
                the assigned label values for each sample.
        """
        assert self.is_labeled

        n_samples = bmus.shape[0]
        assigned_labels = np.empty(shape=(n_samples, self.n_labels))
        for i in range(n_samples):
            row, col = bmus[i]
            assigned_labels[i] = label_maps[:, row, col]

        return assigned_labels


    @property
    def codebook(self) -> np.ndarray:
        assert self.is_trained
        codebook = self.somoclu.codebook
        codebook_scaled_back = self.scaler.inverse_transform(
            codebook.reshape(-1, codebook.shape[2])
        ).reshape(codebook.shape)
        return codebook_scaled_back

