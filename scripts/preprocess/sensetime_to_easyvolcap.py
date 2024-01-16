import untangle
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import write_camera
from easyvolcap.utils.data_utils import as_numpy_func
from easyvolcap.utils.math_utils import affine_inverse


@catch_throw
def main():
    # File paths
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/sensetime4d/wy02')
    parser.add_argument('--xml_file', type=str, default='calib/Calibration.xml')
    parser.add_argument('--scale', type=float, default=0.2)
    args = parser.parse_args()

    xml_file_path = join(args.data_root, args.xml_file)
    tree = untangle.parse(xml_file_path)
    cameras = dotdict()

    # Parse intrinsics
    for sensor in tree.document.chunk.sensors.sensor:
        if sensor.get_attribute('label') != 'unknown':
            name = sensor.get_attribute('label')
            cam = dotdict()
            H, W = int(sensor.resolution.get_attribute('height')), int(sensor.resolution.get_attribute('width'))
            f = float(sensor.calibration.f.cdata)
            cx = (W - 1) / 2 + float(sensor.calibration.cx.cdata)
            cy = (H - 1) / 2 + float(sensor.calibration.cy.cdata)
            K = np.zeros((3, 3))
            K[0, 0] = f
            K[1, 1] = f
            K[0, 2] = cx
            K[1, 2] = cy
            K[2, 2] = 1
            k1 = float(sensor.calibration.k1.cdata)
            k2 = float(sensor.calibration.k2.cdata)
            k3 = float(sensor.calibration.k3.cdata)
            p1 = float(sensor.calibration.p1.cdata)
            p2 = float(sensor.calibration.p2.cdata)
            D = np.asarray([k1, k2, p1, p2, k3])

            cam.H = H
            cam.W = W
            cam.K = K
            cam.D = D
            cameras[name] = cam

    # Parse extrinsics
    for camera in tree.document.chunk.cameras.camera:
        if camera.get_attribute('label') != 'unknown':
            name = camera.get_attribute('label')
            cam = cameras[name]
            c2w = np.asarray(list(map(float, camera.transform.cdata.split(' ')))).reshape(4, 4)
            c2w[:3, 3:4] *= args.scale
            w2c = as_numpy_func(affine_inverse)(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3:]
            cam.R = R
            cam.T = T

    # Reorder cameras
    keys = sorted(cameras.keys())
    cameras = dotdict({f'{i:04d}': cameras[k] for i, k in enumerate(keys)})

    write_camera(cameras, args.data_root)
    log(yellow(f'Converted cameras saved to: {blue(join(args.data_root, "{intri.yml,extri.yml}"))}'))


if __name__ == '__main__':
    main()
