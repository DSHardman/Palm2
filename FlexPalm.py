import matplotlib
import numpy as np
from itertools import count

import serial
import sys
import time

import pyeit.mesh as mesh
import pyeit.eit.protocol as protocol
import pyeit.eit.bp as bp

# Note Amodo Python folder added to 3.12 interpreter path in pycharm settings 01/12/2025
import amodo_eit as eit
from utils import print_info, print_warning, print_error

from vispy import app, scene


# Adapted from live_view.py demo script, dsh46

APP_NAME = "Amodo EIT Live View"
NUM_ELECTRODES = 16  # Change to 16, 32, or 64 as needed
MESH_ELEMENT_SIZE = 0.08  # Smaller = finer mesh
STIM_FREQ_KHZ = 20
PERIODS_PER_MEASUREMENT = 5
TX_GAIN = 32  # 20k, 32 256
RX_GAIN = 256

devices = eit.get_connected_devices()
if not devices:
    print_error("No Amodo EIT devices connected.")
    exit()

device = devices[0]
if len(devices) > 1:
    print_warning("Multiple Amodo EIT devices detected.")

# Setup mesh and protocol
mesh_obj = mesh.create(NUM_ELECTRODES, h0=MESH_ELEMENT_SIZE)
pts = mesh_obj.node
tri = mesh_obj.element

protocol_obj = protocol.create(
    NUM_ELECTRODES,
    dist_exc=1,  # int(NUM_ELECTRODES / 2)
    step_meas=1,
    parser_meas="std"
)

# Setup BP solver
eit_solver = bp.BP(mesh_obj, protocol_obj)
eit_solver.setup(weight="none")

backend = app.use_app("pyqt6")
qapp = backend.native
qapp.setApplicationName(APP_NAME)
qapp.setApplicationDisplayName(APP_NAME)

frame_count = 0
framerate_counter = 0
framerate_counter_start = time.perf_counter()

try:
    with device:
        print_info(f"Using device: {device.port}, version {device.version}, build {device.build_date_time}")

        # device.reset()
        device.set_stimulation_frequency(STIM_FREQ_KHZ)

        # Send electrode configuration
        print_info("Loading electrode configuration...")
        electrode_configurations = []
        # idx = count()
        pin_mapping = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

        for i_exc, exc_pair in enumerate(protocol_obj.ex_mat):
            A, B = exc_pair
            meas_pairs = protocol_obj.meas_mat[i_exc]
            for M, N in meas_pairs:

                configuration = (pin_mapping[A], pin_mapping[B], pin_mapping[M], pin_mapping[N], TX_GAIN, RX_GAIN)  # David 27/11/25

                # if NUM_ELECTRODES == 8:
                #     configuration = (8 * A + 1, 8 * B + 1, 8 * M + 1, 8 * N + 1, TX_GAIN, RX_GAIN)
                # elif NUM_ELECTRODES == 16:
                #     configuration = (4 * A + 1, 4 * B + 1, 4 * M + 1, 4 * N + 1, TX_GAIN, RX_GAIN)
                # elif NUM_ELECTRODES == 32:
                #     configuration = (2 * A + 1, 2 * B + 1, 2 * M + 1, 2 * N + 1, TX_GAIN, RX_GAIN)
                # elif NUM_ELECTRODES == 64:
                #     configuration = (A + 1, B + 1, M + 1, N + 1, TX_GAIN, RX_GAIN)
                # else:
                #     raise RuntimeError("Unsupported number of electrodes")
                electrode_configurations.append(configuration)

        # Swap pairs of electrode configurations:
        # for i in range(0, len(electrode_configurations), 2):
        #     if i + 1 < len(electrode_configurations):
        #         temp = electrode_configurations[i]
        #         electrode_configurations[i] = electrode_configurations[i + 1]
        #         electrode_configurations[i + 1] = temp

        for configuration in electrode_configurations:
            print(f"{configuration[0]} {configuration[1]} {configuration[2]} {configuration[3]} {configuration[4]} {configuration[5]}")

        device.set_electrode_configurations(electrode_configurations)
        print_info(f"Electrode configuration loaded ({len(electrode_configurations)} configurations).\n")

        device.set_num_periods_to_sample_per_measurement(20)
        # device.do_autogain()
        device.set_num_periods_to_sample_per_measurement(PERIODS_PER_MEASUREMENT)

        # Capture baseline frame
        print("Capturing baseline frame...")

        def increment_framerate_counter(data):
            global framerate_counter
            framerate_counter += 1

        device.start_streaming(increment_framerate_counter)

        # while device.latest_frame is None:
        #     time.sleep(0.01)
        # baseline_frame, baseline_clipping = device.latest_frame
        # baseline_frame = np.array([x if x > 1e-12 else 1e-12 for x in baseline_frame])
        # # baseline_frame, baseline_clipping = capture_frame(device._ser)
        # if baseline_clipping:
        #     print_warning("Clipping detected in baseline")

        time.sleep(2)  # First frame is noisier than others

        # Take average baseline rather than single measurement
        baseline, baseline_clipping = device.latest_frame
        baseline = np.array([x if x > 1e-12 else 1e-12 for x in baseline])
        for i in range(9):
            while device.latest_frame is None:
                time.sleep(0.1)
            baseline_frame, baseline_clipping = device.latest_frame
            baseline += np.array([x if x > 1e-12 else 1e-12 for x in baseline_frame])
            # baseline_frame, baseline_clipping = capture_frame(device._ser)
            if baseline_clipping:
                print_warning("Clipping detected in baseline")
        baseline_frame = baseline/10


        print(f"Baseline captured: {len(baseline_frame)} measurements\n")

        ds0 = eit_solver.solve(baseline_frame, baseline_frame, normalize=True)
        vals0 = np.real(ds0)

        n_pts = pts.shape[0]
        n_tri = tri.shape[0]

        def field_to_face(values: np.ndarray) -> np.ndarray:
            """Convert node-based or element-based field to per-face values."""
            if values.shape[0] == n_pts:
                # Node-based values -> average over triangle vertices
                return values[tri].mean(axis=1)
            elif values.shape[0] == n_tri:
                # Already per-face
                return values
            else:
                raise RuntimeError(f"Unexpected field size: {values.shape[0]}, expected {n_pts} (per-vertex) or {n_tri} (per-face)")

        cmap = matplotlib.colormaps["viridis"]

        ds0 = eit_solver.solve(baseline_frame, baseline_frame, normalize=True)
        vals0 = np.real(ds0)

        vals0_face = field_to_face(vals0)  # ensure per-face
        print(f"Using per-face visualization, ds0 size={len(vals0_face)}")

        norm = matplotlib.colors.Normalize(vmin=-1e-12, vmax=1e-12)

        def to_colors(values: np.ndarray) -> np.ndarray:
            """Map scalar field to RGBA float32 using symmetric norm."""
            v = max(np.max(np.abs(values)), 1e-12)
            norm.vmin = -v
            norm.vmax = v
            # norm.vmin = -0.0005
            # norm.vmax = 0.0015
            return cmap(norm(values)).astype(np.float32)

        colors0 = to_colors(vals0_face)

        # ----------------------------------------------------------------
        # VisPy canvas & mesh visual
        # ----------------------------------------------------------------
        canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black", size=(1200, 600), title=APP_NAME)
        vertical_grid = canvas.central_widget.add_grid(margin=0)
        vertical_grid.spacing = 5

        # Mesh view grid:
        mesh_grid = vertical_grid.add_grid(row=0, col=0)
        mesh_grid.spacing = 0

        mesh_view = mesh_grid.add_view(row=0, col=0)
        mesh_view.stretch = (8, 1)
        mesh_view.camera = "panzoom"
        mesh_view.camera.set_range(x=(-1.0, 1.0), y=(-1.1, 1.1))
        mesh_view.camera.aspect = 1.0

        # Colorbar on the right
        cbar = scene.ColorBarWidget(
            cmap="viridis",
            orientation="left",
            border_width=0.0,
            label_color="white",
            padding=(0.05, 0.3),
        )
        for t in cbar.ticks:
            t.font_size = 8
        mesh_grid.add_widget(cbar, row=0, col=1)

        line_grid = vertical_grid.add_grid(row=1, col=0, spacing=0, margin=0)
        line_view = line_grid.add_view(row=0, col=1, margin=0, padding=0)
        line_view.camera = scene.cameras.PanZoomCamera(aspect=None, interactive=False)

        yaxis = scene.AxisWidget(
            orientation="left",
            tick_font_size=6,
            tick_label_margin=5,
            axis_width=1,
        )
        yaxis.width_max = 60
        line_grid.add_widget(yaxis, row=0, col=0)
        yaxis.link_view(line_view)

        # Create line plot
        n_measurements = len(baseline_frame)
        line_x = np.arange(n_measurements)
        line_y = baseline_frame
        line_plot = scene.visuals.Line(pos=np.column_stack((line_x, line_y)), color="cyan", width=1, parent=line_view.scene)

        mesh_visual = scene.visuals.Mesh(
            vertices=mesh_obj.node,
            faces=tri,
            face_colors=colors0,
            shading=None,
            parent=mesh_view.scene,
        )

        # Text overlay for frame info
        text = scene.Label("Frame 0", color="white", font_size=10, anchor_x="left", anchor_y="bottom")
        text.parent = canvas.scene  # set parent afterwards
        text.pos = (5, 5)

        def on_key_press(event):
            global baseline_frame
            if event.key == "R":
                current_frame, _ = device.latest_frame
                if current_frame is not None:
                    baseline_frame = current_frame
                    print_info("Baseline frame reset!")
                    text.text = "Baseline RESET!"

        canvas.connect(on_key_press)

        # ----------------------------------------------------------------
        # Update callback: called repeatedly by VisPy's Timer
        # ----------------------------------------------------------------
        def update(event):
            current_frame, clipping = device.latest_frame

            # Update line plot
            line_y = current_frame
            line_pos = np.column_stack((line_x, line_y))
            line_plot.set_data(pos=line_pos)

            pos = line_pos  # or whatever array you're using
            x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
            y_min, y_max = pos[:, 1].min(), pos[:, 1].max()

            # Make sure aspect is free to avoid squashing x when y grows
            line_view.camera.aspect = None
            y_range = y_max - y_min
            line_view.camera.set_range(x=(x_min, x_max), y=(y_min - y_range * 0.05, y_max + y_range * 0.05), margin=1e-4)

            # Ensure no zeros in current_frame:
            current_frame = np.array([x if x > 1e-12 else 1e-12 for x in current_frame])
            ds = eit_solver.solve(current_frame, baseline_frame, normalize=True)
            vals = np.real(ds)

            vals_face = field_to_face(vals)
            colors = to_colors(vals_face)

            cbar.clim = (round(norm.vmin, 2), round(norm.vmax, 2))

            if colors.shape[0] != tri.shape[0]:
                print_error("Color/face count mismatch")
                timer.stop()
                app.quit()
                return
            mesh_visual.set_data(
                vertices=mesh_obj.node,
                faces=tri,
                face_colors=colors,
            )

            global framerate_counter, framerate_counter_start
            if time.perf_counter() - framerate_counter_start >= 1.0:
                text.text = (
                    f"Frame: ({framerate_counter:.1f} Hz) - {'CLIPPING!' if clipping else 'OK'} - FREQ: {STIM_FREQ_KHZ} kHz, PERIODS: {PERIODS_PER_MEASUREMENT}, TX GAIN: {TX_GAIN}, RX GAIN: {RX_GAIN}"
                )
                print_info(f"Framerate: {framerate_counter} FPS")
                framerate_counter = 0
                framerate_counter_start = time.perf_counter()

        timer = app.Timer(interval=1 / 60.0, connect=update, start=True)

        print("Starting VisPy live view (close window to exit)...")
        app.run()

except KeyboardInterrupt:
    print("\n\nStopped by user")

except serial.SerialException as e:
    print_error(f"Serial error: {e}")
    sys.exit(1)

except Exception as e:
    print_error(f"Unexpected error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("Done.")
