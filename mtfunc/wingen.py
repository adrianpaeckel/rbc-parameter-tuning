import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                train_df, val_df, test_df,example_df=None,label_columns=None,input_columns=None,stride=None):
        self.label_columns = label_columns
        self.input_columns=input_columns
        # Store the raw data.
        self.stride=stride
        self.train_df = train_df[list(set(self.input_columns)|set(self.label_columns))]
        self.val_df = val_df[list(set(self.input_columns)|set(self.label_columns))]
        self.test_df = test_df[list(set(self.input_columns)|set(self.label_columns))]
        self.example_df=example_df
        
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        if input_columns is not None:
            self.input_columns_indices = {name: i for i, name in
                                        enumerate(input_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
    
    def plot(self, model=None, plot_col='dsoc',plot_input='soc_fil', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_input_index = self.input_columns_indices[plot_input]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_input_index],
                    label='Inputs', marker='.', zorder=-10)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_input_index
            if label_col_index is None:
                continue
            plt.scatter(self.label_indices,inputs[n, :, plot_input_index]+labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=50)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, inputs[n, :, plot_input_index]+predictions[n, :, label_col_index],
                        marker='X', edgecolors='b', label='Predictions',
                        c='#ff7f0e', s=50)
#             plt.ylim([-1,1])

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')    
        
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        if self.input_columns is not None:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]] for name in self.input_columns],
                axis=-1)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])


        return inputs, labels

    def make_dataset(self, data,batch_size=32,shuffle=True):
#         print(data)
        data = np.array(data, dtype=np.float32)
#         print(data.shape)
        if self.stride is not None: st=self.stride
        else: st=1
            
            
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=st,

        shuffle=shuffle,
        batch_size=batch_size,)

        ds = ds.map(self.split_window)
#         print(ds)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
#       """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, 'example_df', None)
        if result is None:
            self._example = self.train
        else: 
            self._example=self.make_dataset(self.example_df,batch_size=3,shuffle=True)
        return next(iter(self._example))