import numpy as np
import re


class DataLoader:
    NORMED_DETECTOR_EFFICIENCY = 3.24e4
    NORMED_DETECTOR_WAVELENGTH = 1.8e-10  # m
    INCIDENT_WAVE_VECTOR = 1.4e10  # m-1

    PARAMETER_SUFFIX = '_value'
    DATA_START = 'Scan data'
    COUNT_TIME = 'det_preset'
    SCAN_INFO = 'info'

    PLOT_Y_LABEL = 'Normalised counts'

    # Parameter names of the current

    DCT = 'dct'
    DCT1 = 'dct1'
    DCT2 = 'dct2'
    DCT3 = 'dct3'
    DCT4 = 'dct4'
    DCT5 = 'dct5'
    DCT6 = 'dct6'
    DCTs = [DCT1, DCT2, DCT3, DCT4, DCT5, DCT6]

    SCATTERING_ANGLE = 'stt'

    # Parameter names of the sample table
    SAMPLE_TABLE_ANGLE = 'sth_st'
    SAMPLE_TABLE_POSITION = ['stx', 'sty', 'stz']
    SAMPLE_TABLE_TILT = ['sgx', 'sgy']
    SAMPLE_TABLE = [SAMPLE_TABLE_ANGLE] + SAMPLE_TABLE_POSITION + SAMPLE_TABLE_TILT

    # Including two pieces of information, i.e. width and height
    SLIT1 = 'ss1'  # (centre_x, centre_y) width x height mm
    SLIT2 = 'ss2'  # but we do not need it at the moment
    SLITs = [SLIT1, SLIT2]

    SLIT_INFO_1 = 'centre_x'
    SLIT_INFO_2 = 'centre_y'
    SLIT_INFO_3 = 'width'
    SLIT_INFO_4 = 'height'
    SLIT_INFOs = [SLIT_INFO_1, SLIT_INFO_2, SLIT_INFO_3, SLIT_INFO_4]

    SLITS_INFOS = []
    for slit in SLITs:
        for slit_info in SLIT_INFOs:
            SLITS_INFOS.append(str(slit) + '_' + str(slit_info))

    # With a form: xx_value : $float unit
    NORMAL_FLOAT_VALUES = DCTs + SAMPLE_TABLE + [SCATTERING_ANGLE]

    INFOS_NEEDED = [SCAN_INFO] + NORMAL_FLOAT_VALUES + SLITS_INFOS

    def __init__(self, file_index, file_prefix='15315_000', file_format='dat'):
        self.file_prefix = file_prefix
        self.file_format = file_format
        self.filename = self.file_prefix + str(file_index) + self.file_format

        self.data_2d_array = True
        self.x_data = None
        self.real_counts = None

        self.__properties = {}
        self.properties_sorted = {}

    def data_getter(self):
        # reads only the data part of a file
        # x-axis	timer	mon1	mon2	ctr1    ctr2
        try:
            data = np.loadtxt(self.filename, comments='#')

            if data.ndim == 2:
                self.data_2d_array = True
                self.x_data = data[:, 0]
                counts = data[:, -2]
                monitor = data[:, -3]
                self.real_counts = counts / self._normalised_counts(monitor)
                self.real_counts = np.int_(self.real_counts)
                self._raw_data_correction()
            else:
                self.data_2d_array = False
        except UserWarning:
            self.data_2d_array = False

    def _raw_data_correction(self):
        error = 1e-3
        data_length = self.x_data.shape[0]
        delete_index = []
        for i in range(data_length):
            # one point was shifted to the next
            if 0 < i < data_length - 2 and self.x_data[i] > self.x_data[i - 1] and abs(
                    self.x_data[i] - self.x_data[i + 1]) < error:
                self.x_data[i] = (self.x_data[i - 1] + self.x_data[i + 1]) / 2.0
            # zero count
            if int(self.real_counts[i]) == 0:
                delete_index.append(i)

        self.x_data = np.delete(self.x_data, delete_index)
        self.real_counts = np.delete(self.real_counts, delete_index)
        # self.monitor = np.delete(self.monitor, delete_index)
        if self.x_data.shape[0] == 0:
            self.data_2d_array = False
        else:
            # first point was shifted to another position
            if self.x_data[0] > self.x_data[1]:
                self.x_data[0] = 0

    @staticmethod
    def _unit_correction(unit):
        if unit == 'A':
            pass
        else:
            unit = 'A'
        return unit

    def _normalised_counts(self, mon_counts):
        wavelength = 2 * np.pi / float(self.INCIDENT_WAVE_VECTOR)
        flux = mon_counts / self.NORMED_DETECTOR_EFFICIENCY * wavelength / self.NORMED_DETECTOR_WAVELENGTH
        return flux

    def _parse_header(self, data_line):
        # parses a pair of floats: "($float, $float)"
        def slit_matcher(s):
            regex = r'\(\s*([-+]?[0-9]*\.?[0-9]*)\s*,\s*([-+]?[0-9]*\.?[0-9]*\s*)\)\s*' \
                    r'(\s*[-+]?[0-9]*\.?[0-9]*)\s*x\s*([-+]?[0-9]*\.?[0-9]*)\s*'
            try:
                match = re.search(regex, s, re.IGNORECASE)
                return list(map(float, [match.group(1), match.group(2), match.group(3), match.group(4)]))
            except TypeError:
                print("Wrong data type of pair matching")
            except:
                print("Function does not work: slit_matcher")

        # parses a float, returns the first float from the left, anything else is ignored
        def float_matcher(s):
            regex = r'\s*([-+]?[0-9]*\.?[0-9]*)\s*'
            try:
                match = re.search(regex, s, re.IGNORECASE)
                return float(match.group(1))
            except TypeError:
                print("Wrong data type of float matching")
            except OSError:
                pass

        # only the lines with colon in between are needed
        if ':' in data_line:
            # seems like we have a meta-data in the form "key : value"
            # we have to be careful, since raw_data can have multiple entries (more than 2), because
            # somebody thought it is a good idea to have a separator in the data section..
            # for example: "ms2_status : ok: left_idle ...
            key, value = list(map(lambda s: s.strip(), data_line[1:].split(':', maxsplit=1)))
            if self.PARAMETER_SUFFIX in key:
                key = key.replace(self.PARAMETER_SUFFIX, '')
                if key in self.NORMAL_FLOAT_VALUES:
                    value = float_matcher(value)
                    return key, value
                if key in self.SLITs:
                    values = slit_matcher(value)
                    slit_infos = list(map(lambda s: key + '_' + s, self.SLIT_INFOs))
                    return slit_infos, values
            elif self.COUNT_TIME in key or self.SCAN_INFO in key:
                return key, value

    def _scan_variable(self, line1, line2):
        scan_name, unit = list(map(lambda s: s.split()[1], (line1, line2)))
        if scan_name in self.DCTs:
            unit = self._unit_correction(unit)
        return scan_name

    # grabs __properties from a file that are encoded in the form of "key: value"
    # Reforming realised by means of parsing funcs
    def _property_raw(self):
        f = open(self.filename, "r")
        lines = f.readlines()
        f.close()

        scan_key = None
        for line in lines:
            if line.startswith('#'):
                if line.startswith('###'):
                    if self.DATA_START in line:
                        data_start_index = lines.index(line)
                        scan_key = self._scan_variable(lines[data_start_index + 1], lines[data_start_index + 2])
                try:
                    key, value = self._parse_header(line)
                except:
                    continue
                if isinstance(key, str):
                    self.__properties.update({key: value})
                elif isinstance(key, list):
                    self.__properties.update(dict(zip(key, value)))
                # except ValueError:
                #     print(self._parse_header(line))
        self.__properties[scan_key] = 'Scanned'

    def property_getter(self):
        self._property_raw()
        for key in self.INFOS_NEEDED:
            try:
                self.properties_sorted.update({key: self.__properties[key]})
            except KeyError:  # for optional infos
                self.properties_sorted.update({key: 'Unused'})
