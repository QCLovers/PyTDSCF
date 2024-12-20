import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.typing import NDArray


class ComplexMatrixAnimation:
    fig: plt.Figure
    ax: plt.Axes
    cax: plt.Axes

    def __init__(
        self,
        data: NDArray[np.complex128],
        time: NDArray | None = None,
        title: str = "Complex Matrix Hinton Plot",
        row_names: list[str] | None = None,
        col_names: list[str] | None = None,
        time_unit: str = "fs",
        cmap: str = "hsv",
        figshape: tuple[int, int] = (14, 10),
    ) -> None:
        self.data = data
        if time is None:
            time = np.arange(data.shape[0], dtype=np.float64)
        self.time = time
        self.title = title
        self.row_names = row_names
        self.col_names = col_names
        self.time_unit = time_unit
        self.figshape = figshape
        self.cmap = cmap
        self._validate_input()
        self.norm = np.abs(self.data).real
        self.maxnorm = self.norm.max()
        phase = np.angle(self.data)  # -pi to pi
        self.phase = (phase + 2 * np.pi) % (2 * np.pi)  # 0 to 2pi

    @property
    def rows(self) -> int:
        return self.data.shape[1]

    @property
    def cols(self) -> int:
        return self.data.shape[2]

    def _validate_input(self) -> None:
        """Validate input data for complex matrix animation.

        Raises:
            ValueError: If data is not complex128 or not a square matrix
        """
        if (
            not isinstance(self.data, np.ndarray)
            or self.data.dtype != np.complex128
        ):
            raise ValueError("Input must be a complex128 numpy array")

        if not isinstance(self.time, np.ndarray):
            raise ValueError("Time must be a numpy array")

        if len(self.data.shape) != 3:
            raise ValueError("Input must have shape (time, row, column)")

        if len(self.time.shape) != 1:
            raise ValueError("Time must be a 1D array")

        if self.data.shape[0] != self.time.shape[0]:
            raise ValueError(
                "Time steps must match the first dimension of the data"
            )

        if (
            self.row_names is not None
            and len(self.row_names) != self.data.shape[1]
        ):
            raise ValueError(
                "Number of row names must match the number of rows in the matrix"
            )
        if (
            self.col_names is not None
            and len(self.col_names) != self.data.shape[2]
        ):
            raise ValueError(
                "Number of column names must match the number of columns in the matrix"
            )

        _, rows, cols = self.data.shape
        if rows != cols:
            raise ValueError(
                f"Each frame must be a square matrix, got shape ({rows}, {cols})"
            )

    def set_ax(self, title: str | None = None) -> None:
        ax = self.ax
        cols = self.cols
        rows = self.rows
        if title is None:
            title = self.title
        assert isinstance(title, str)
        row_names = self.row_names
        col_names = self.col_names
        ax.set_title(title, fontsize=24)
        ax.set_xlim(-1, cols)
        ax.set_ylim(-1, rows)
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.invert_yaxis()
        if row_names:
            # set fontsize=14
            ax.set_yticklabels(row_names, fontsize=16)
        if col_names:
            ax.set_xticklabels(col_names, fontsize=16)

    def setup_figure(
        self,
    ):
        """Set up the figure and axes for the Hinton plot.

        Args:
            shape: Shape of the matrix (rows, columns)

        Returns:
            Figure and Axes objects
        """
        plt.ioff()
        self.fig = plt.figure(figsize=self.figshape)
        self.ax = plt.axes(
            (0.1, 0.1, 0.7, 0.8)
        )  # [left, bottom, width, height]
        self.set_ax()
        self.cax = plt.axes((0.75, 0.4, 0.2, 0.2), projection="polar")
        self.set_cyclic_colorbar()

    def plot_each_element(
        self,
        i: int,
        j: int,
        cmap: plt.Colormap,
        norm: np.ndarray,
        phase: np.ndarray,
        data: np.ndarray,
    ) -> None:
        """Plot each element of the complex matrix.

        Args:
            i: Row index
            j: Column index
            cmap: Colormap object
            norm: Magnitude of the complex number
            phase: Phase of the complex number
            data: Complex matrix data
        """
        magnitude = norm[i, j]
        phase_value = phase[i, j]
        value = data[i, j]

        if magnitude > 0:
            # Size based on normalized magnitude
            size = (magnitude / self.maxnorm) * 0.95

            # Color based on phase (normalize from [-π, π] to [0, 1])
            color = cmap(phase_value / (2 * np.pi))

            # Create and add rectangle
            rect = Rectangle(
                (j - size / 2, i - size / 2),
                size,
                size,
                facecolor=color,
                edgecolor="gray",
            )
            self.ax.add_patch(rect)

            # Add text annotation
            text = f"{value: .2f}"
            self.ax.text(
                j,
                i,
                text,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"),
            )

    def update(self, frame_num: int) -> None:
        """Update function for animation.

        Args:
            frame_num: Frame number
        """
        # Get current frame data
        frame_data = self.data[frame_num]
        rows, cols = frame_data.shape
        time = self.time[frame_num]
        title = f"{self.title} {time: .2f} {self.time_unit}"
        self.ax.clear()
        self.set_ax(title)
        _cmap = plt.get_cmap(self.cmap)
        norm = self.norm[frame_num]
        phase = self.phase[frame_num]

        # Plot each element
        for i in range(rows):
            for j in range(cols):
                self.plot_each_element(i, j, _cmap, norm, phase, frame_data)

    def set_cyclic_colorbar(self) -> mcolors.Colormap:
        ax = self.cax

        theta = np.linspace(0.0, 2 * np.pi, 100)
        r = np.linspace(0, 1, 100)

        Theta, R = np.meshgrid(theta, r)

        cmap = plt.get_cmap(self.cmap)
        norm = mcolors.Normalize(vmin=0.0, vmax=2 * np.pi)

        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels(
            ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$"],
            fontsize=14,
        )
        ax.set_yticks([])
        ax.pcolormesh(
            Theta, R, Theta, cmap=cmap, norm=norm
        )  # , shading="auto", alpha=R / R.max())
        return cmap

    def create_animation(
        self,
        interval: int = 200,
    ) -> tuple[plt.Figure, animation.FuncAnimation]:
        """Create an animation of complex matrix Hinton plots.

        Args:
            data: Complex array of shape (time, row, column)
            interval: Time interval between frames in milliseconds

        Returns:
            Figure and Animation objects
        """
        self.setup_figure()

        # Create animation
        anim = animation.FuncAnimation(
            self.fig,
            self.update,  # type: ignore
            frames=self.data.shape[0],
            fargs=(),
            interval=interval,
            blit=False,
        )
        return self.fig, anim


def save_animation(
    anim: animation.FuncAnimation,
    filename: str = "animation.gif",
    fps: int = 5,
    dpi: int = 100,
) -> None:
    """Save animation as a GIF file.

    Args:
        anim: Animation object
        filename: Output filename
        fps: Frames per second
        dpi: Dots per inch for the output
    """
    print(f"Saving animation to {filename}...")
    writer = animation.PillowWriter(fps=fps)
    anim.save(filename, writer=writer, dpi=dpi)
    print("Animation saved successfully!")


def get_anim(
    data: NDArray[np.complex128],
    time: NDArray | None = None,
    title: str = "Density Matrix Evolution",
    row_names: list[str] | None = None,
    col_names: list[str] | None = None,
    time_unit: str = "fs",
    save_gif: bool = False,
    gif_filename: str = "animation.gif",
    cmap: str = "hsv",
    fps: int = 5,
    dpi: int = 100,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """Main function to create Hinton plot animation from complex matrix data.

    Args:
        data (NDArray[np.complex128]): Complex array of shape (time, row, column).
        time (NDArray | None, optional): Array of time points corresponding to the data. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "Complex Matrix Hinton Plot".
        row_names (list[str] | None, optional): List of row names. Defaults to None.
        col_names (list[str] | None, optional): List of column names. Defaults to None.
        time_unit (str, optional): Unit of time to display on the plot. Defaults to "".
        save_gif (bool, optional): Whether to save the animation as a GIF. Defaults to False.
        gif_filename (str, optional): Output filename for GIF. Defaults to "animation.gif".
        cmap (str, optional): Colormap to use for the plot. Defaults to "hsv".
            `cmap` should be cyclic such as 'twilight', 'twilight_shifted', 'hsv'.
            See also https://matplotlib.org/stable/users/explain/colors/colormaps.html#cyclic.
        fps (int, optional): Frames per second for GIF. Defaults to 5.
        dpi (int, optional): Dots per inch for the output GIF. Defaults to 100.


    Returns:
        tuple[plt.Figure, animation.FuncAnimation]: Figure and Animation objects.

    Example:
        >>> # Create a 3x3 complex matrix that evolves over 10 time steps
        >>> t = np.linspace(0, 2*np.pi, 10)
        >>> data = np.zeros((10, 3, 3), dtype=np.complex128)
        >>> for i in range(10):
        ...     data[i] = np.exp(1j * t[i]) * np.random.random((3, 3))
        >>> fig, anim = main(data, time=t, save_gif=True)
        >>> plt.show()
    """
    # Create animation object
    anim_obj = ComplexMatrixAnimation(
        data, time, title, row_names, col_names, time_unit, cmap=cmap
    )

    # Create animation
    fig, anim = anim_obj.create_animation()

    if save_gif:
        save_animation(anim, gif_filename, fps, dpi)

    return fig, anim


if __name__ == "__main__":
    # Create example data: rotating complex numbers
    time_steps = 20
    size = 5
    t = np.linspace(0, 2 * np.pi, time_steps)

    # Initialize complex matrix
    test_data = np.zeros((time_steps, size, size), dtype=np.complex128)

    # Create rotating complex numbers with varying magnitudes
    for i in range(time_steps):
        magnitude = (
            np.random.random((size, size)) + 0.5
        )  # Random magnitudes > 0.5
        phase = (
            t[i] + np.random.random((size, size)) * np.pi / 4
        )  # Base rotation + noise
        test_data[i] = magnitude * np.exp(1j * phase)

    # Create animation and save as GIF
    print("Creating animation...")
    fig, anim = get_anim(
        test_data,
        time=t,
        title="Density Matrix Evolution",
        save_gif=True,
        gif_filename="complex_matrix.gif",
        cmap="twilight_shifted",
        time_unit="fs",
        row_names=["$|" + f"{i}" + r"\rangle$" for i in range(size)],
        col_names=[r"$\langle" + f"{i}" + r"|$" for i in range(size)],
        fps=5,
        dpi=100,
    )
    plt.show()
