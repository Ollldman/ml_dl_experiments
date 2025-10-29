import matplotlib.pyplot as plt
from pandas import DataFrame

def plot_time_series(time_series_df: DataFrame,
                     plot_title: str,
                     plot_label: str,
                     name_column_one: str | None,
                     name_column_two: str,
                     labels_x_y: list[str])-> None:
    plt.figure(figsize=(10, 4))
    if name_column_one:
        plt.plot(time_series_df[name_column_one],
                time_series_df[name_column_two],
                label=plot_label)
    else:
        plt.plot(time_series_df.index,
                time_series_df[name_column_two],
                label=plot_label)
    plt.xlabel(labels_x_y[0])
    plt.ylabel(labels_x_y[1])
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.show()