from pydicom.tag import BaseTag
from pydicom.multival import MultiValue
from FilmLab.dicom import PythonFileAutoDicom, PythonAutoDicom, PythonImageFileAutoDicom

class MVEPIDImageAutoDicom(PythonImageFileAutoDicom):
    description = "MV EPID Dicom from TrueBeam"
    @property
    def sop_instance_uid(self):
        if (0x0008,0x0018) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0018].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0018].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0018].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0018].value]
            else:
                return self._pydicom_obj[0x0008,0x0018].value
        else:
            return None

    @property
    def isocenter_position(self):
        if (0x300a,0x012c) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x012c].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x012c].value
            elif (isinstance(self._pydicom_obj[0x300a,0x012c].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300a,0x012c].value]
            else:
                return float(self._pydicom_obj[0x300a,0x012c].value)
        else:
            return None

    @property
    def study_time(self):
        if (0x0008,0x0030) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0030].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0030].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0030].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0030].value]
            else:
                return self._pydicom_obj[0x0008,0x0030].value
        else:
            return None

    @property
    def data_value_representation(self):
        if (0x5000,0x0103) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x5000,0x0103].value, BaseTag)):
                return self._pydicom_obj[0x5000,0x0103].value
            elif (isinstance(self._pydicom_obj[0x5000,0x0103].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x5000,0x0103].value]
            else:
                return self._pydicom_obj[0x5000,0x0103].value
        else:
            return None

    @property
    def rt_image_position(self):
        if (0x3002,0x0012) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3002,0x0012].value, BaseTag)):
                return self._pydicom_obj[0x3002,0x0012].value
            elif (isinstance(self._pydicom_obj[0x3002,0x0012].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x3002,0x0012].value]
            else:
                return float(self._pydicom_obj[0x3002,0x0012].value)
        else:
            return None

    @property
    def modality(self):
        if (0x0008,0x0060) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0060].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0060].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0060].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0060].value]
            else:
                return self._pydicom_obj[0x0008,0x0060].value
        else:
            return None

    @property
    def patient_id(self):
        if (0x0010,0x0020) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0010,0x0020].value, BaseTag)):
                return self._pydicom_obj[0x0010,0x0020].value
            elif (isinstance(self._pydicom_obj[0x0010,0x0020].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0010,0x0020].value]
            else:
                return self._pydicom_obj[0x0010,0x0020].value
        else:
            return None

    @property
    def rescale_slope(self):
        if (0x0028,0x1053) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x1053].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x1053].value
            elif (isinstance(self._pydicom_obj[0x0028,0x1053].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x0028,0x1053].value]
            else:
                return float(self._pydicom_obj[0x0028,0x1053].value)
        else:
            return None

    @property
    def referenced_beam_number(self):
        if (0x300c,0x0006) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300c,0x0006].value, BaseTag)):
                return self._pydicom_obj[0x300c,0x0006].value
            elif (isinstance(self._pydicom_obj[0x300c,0x0006].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x300c,0x0006].value]
            else:
                return self._pydicom_obj[0x300c,0x0006].value
        else:
            return None

    @property
    def table_top_lateral_position(self):
        if (0x300a,0x012a) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x012a].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x012a].value
            elif (isinstance(self._pydicom_obj[0x300a,0x012a].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300a,0x012a].value]
            else:
                return float(self._pydicom_obj[0x300a,0x012a].value)
        else:
            return None

    @property
    def irradiation_event_uid(self):
        if (0x0008,0x3010) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x3010].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x3010].value
            elif (isinstance(self._pydicom_obj[0x0008,0x3010].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x3010].value]
            else:
                return self._pydicom_obj[0x0008,0x3010].value
        else:
            return None

    @property
    def acquisition_time(self):
        if (0x0008,0x0032) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0032].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0032].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0032].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0032].value]
            else:
                return self._pydicom_obj[0x0008,0x0032].value
        else:
            return None

    @property
    def frame_of_reference_uid(self):
        if (0x0020,0x0052) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0020,0x0052].value, BaseTag)):
                return self._pydicom_obj[0x0020,0x0052].value
            elif (isinstance(self._pydicom_obj[0x0020,0x0052].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0020,0x0052].value]
            else:
                return self._pydicom_obj[0x0020,0x0052].value
        else:
            return None

    @property
    def beam_limiting_device_angle(self):
        if (0x300a,0x0120) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x0120].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x0120].value
            elif (isinstance(self._pydicom_obj[0x300a,0x0120].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300a,0x0120].value]
            else:
                return float(self._pydicom_obj[0x300a,0x0120].value)
        else:
            return None

    @property
    def start_cumulative_meterset_weight(self):
        if (0x300c,0x0008) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300c,0x0008].value, BaseTag)):
                return self._pydicom_obj[0x300c,0x0008].value
            elif (isinstance(self._pydicom_obj[0x300c,0x0008].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300c,0x0008].value]
            else:
                return float(self._pydicom_obj[0x300c,0x0008].value)
        else:
            return None

    @property
    def sop_class_uid(self):
        if (0x0008,0x0016) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0016].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0016].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0016].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0016].value]
            else:
                return self._pydicom_obj[0x0008,0x0016].value
        else:
            return None

    @property
    def referring_physicians_name(self):
        if (0x0008,0x0090) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0090].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0090].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0090].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0090].value]
            else:
                return self._pydicom_obj[0x0008,0x0090].value
        else:
            return None

    @property
    def rt_image_label(self):
        if (0x3002,0x0002) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3002,0x0002].value, BaseTag)):
                return self._pydicom_obj[0x3002,0x0002].value
            elif (isinstance(self._pydicom_obj[0x3002,0x0002].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x3002,0x0002].value]
            else:
                return self._pydicom_obj[0x3002,0x0002].value
        else:
            return None

    @property
    def patients_sex(self):
        if (0x0010,0x0040) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0010,0x0040].value, BaseTag)):
                return self._pydicom_obj[0x0010,0x0040].value
            elif (isinstance(self._pydicom_obj[0x0010,0x0040].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0010,0x0040].value]
            else:
                return self._pydicom_obj[0x0010,0x0040].value
        else:
            return None

    @property
    def number_of_points(self):
        if (0x5000,0x0010) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x5000,0x0010].value, BaseTag)):
                return self._pydicom_obj[0x5000,0x0010].value
            elif (isinstance(self._pydicom_obj[0x5000,0x0010].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x5000,0x0010].value]
            else:
                return self._pydicom_obj[0x5000,0x0010].value
        else:
            return None

    @property
    def pixel_representation(self):
        if (0x0028,0x0103) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x0103].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x0103].value
            elif (isinstance(self._pydicom_obj[0x0028,0x0103].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0028,0x0103].value]
            else:
                return self._pydicom_obj[0x0028,0x0103].value
        else:
            return None

    @property
    def curve_data(self):
        if (0x5000,0x3000) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x5000,0x3000].value, BaseTag)):
                return self._pydicom_obj[0x5000,0x3000].value
            elif (isinstance(self._pydicom_obj[0x5000,0x3000].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x5000,0x3000].value]
            else:
                return self._pydicom_obj[0x5000,0x3000].value
        else:
            return None

    @property
    def end_cumulative_meterset_weight(self):
        if (0x300c,0x0009) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300c,0x0009].value, BaseTag)):
                return self._pydicom_obj[0x300c,0x0009].value
            elif (isinstance(self._pydicom_obj[0x300c,0x0009].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300c,0x0009].value]
            else:
                return float(self._pydicom_obj[0x300c,0x0009].value)
        else:
            return None

    @property
    def device_serial_number(self):
        if (0x0018,0x1000) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0018,0x1000].value, BaseTag)):
                return self._pydicom_obj[0x0018,0x1000].value
            elif (isinstance(self._pydicom_obj[0x0018,0x1000].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0018,0x1000].value]
            else:
                return self._pydicom_obj[0x0018,0x1000].value
        else:
            return None

    @property
    def curve_label(self):
        if (0x5000,0x2500) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x5000,0x2500].value, BaseTag)):
                return self._pydicom_obj[0x5000,0x2500].value
            elif (isinstance(self._pydicom_obj[0x5000,0x2500].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x5000,0x2500].value]
            else:
                return self._pydicom_obj[0x5000,0x2500].value
        else:
            return None

    @property
    def station_name(self):
        if (0x0008,0x1010) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x1010].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x1010].value
            elif (isinstance(self._pydicom_obj[0x0008,0x1010].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x1010].value]
            else:
                return self._pydicom_obj[0x0008,0x1010].value
        else:
            return None

    @property
    def position_reference_indicator(self):
        if (0x0020,0x1040) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0020,0x1040].value, BaseTag)):
                return self._pydicom_obj[0x0020,0x1040].value
            elif (isinstance(self._pydicom_obj[0x0020,0x1040].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0020,0x1040].value]
            else:
                return self._pydicom_obj[0x0020,0x1040].value
        else:
            return None

    @property
    def study_date(self):
        if (0x0008,0x0020) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0020].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0020].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0020].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0020].value]
            else:
                return self._pydicom_obj[0x0008,0x0020].value
        else:
            return None

    @property
    def operators_name(self):
        if (0x0008,0x1070) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x1070].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x1070].value
            elif (isinstance(self._pydicom_obj[0x0008,0x1070].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x1070].value]
            else:
                return self._pydicom_obj[0x0008,0x1070].value
        else:
            return None

    @property
    def extended_interface_format_tag(self):
        if (0x3253,0x1002) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3253,0x1002].value, BaseTag)):
                return self._pydicom_obj[0x3253,0x1002].value
            elif (isinstance(self._pydicom_obj[0x3253,0x1002].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x3253,0x1002].value]
            else:
                return self._pydicom_obj[0x3253,0x1002].value
        else:
            return None

    @property
    def study_instance_uid(self):
        if (0x0020,0x000d) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0020,0x000d].value, BaseTag)):
                return self._pydicom_obj[0x0020,0x000d].value
            elif (isinstance(self._pydicom_obj[0x0020,0x000d].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0020,0x000d].value]
            else:
                return self._pydicom_obj[0x0020,0x000d].value
        else:
            return None

    @property
    def acquisition_date(self):
        if (0x0008,0x0022) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0022].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0022].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0022].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0022].value]
            else:
                return self._pydicom_obj[0x0008,0x0022].value
        else:
            return None

    @property
    def specific_character_set(self):
        if (0x0008,0x0005) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0005].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0005].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0005].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0005].value]
            else:
                return self._pydicom_obj[0x0008,0x0005].value
        else:
            return None

    @property
    def content_time(self):
        if (0x0008,0x0033) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0033].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0033].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0033].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0033].value]
            else:
                return self._pydicom_obj[0x0008,0x0033].value
        else:
            return None

    @property
    def window_width(self):
        if (0x0028,0x1051) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x1051].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x1051].value
            elif (isinstance(self._pydicom_obj[0x0028,0x1051].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x0028,0x1051].value]
            else:
                return float(self._pydicom_obj[0x0028,0x1051].value)
        else:
            return None

    @property
    def software_versions(self):
        if (0x0018,0x1020) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0018,0x1020].value, BaseTag)):
                return self._pydicom_obj[0x0018,0x1020].value
            elif (isinstance(self._pydicom_obj[0x0018,0x1020].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0018,0x1020].value]
            else:
                return self._pydicom_obj[0x0018,0x1020].value
        else:
            return None

    @property
    def axis_units(self):
        if (0x5000,0x0030) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x5000,0x0030].value, BaseTag)):
                return self._pydicom_obj[0x5000,0x0030].value
            elif (isinstance(self._pydicom_obj[0x5000,0x0030].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x5000,0x0030].value]
            else:
                return self._pydicom_obj[0x5000,0x0030].value
        else:
            return None

    @property
    def high_bit(self):
        if (0x0028,0x0102) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x0102].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x0102].value
            elif (isinstance(self._pydicom_obj[0x0028,0x0102].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0028,0x0102].value]
            else:
                return self._pydicom_obj[0x0028,0x0102].value
        else:
            return None

    @property
    def rt_image_sid(self):
        if (0x3002,0x0026) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3002,0x0026].value, BaseTag)):
                return self._pydicom_obj[0x3002,0x0026].value
            elif (isinstance(self._pydicom_obj[0x3002,0x0026].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x3002,0x0026].value]
            else:
                return float(self._pydicom_obj[0x3002,0x0026].value)
        else:
            return None

    @property
    def image_type(self):
        if (0x0008,0x0008) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0008].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0008].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0008].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0008].value]
            else:
                return self._pydicom_obj[0x0008,0x0008].value
        else:
            return None

    @property
    def pixel_intensity_relationship(self):
        if (0x0028,0x1040) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x1040].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x1040].value
            elif (isinstance(self._pydicom_obj[0x0028,0x1040].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0028,0x1040].value]
            else:
                return self._pydicom_obj[0x0028,0x1040].value
        else:
            return None

    @property
    def columns(self):
        if (0x0028,0x0011) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x0011].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x0011].value
            elif (isinstance(self._pydicom_obj[0x0028,0x0011].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0028,0x0011].value]
            else:
                return self._pydicom_obj[0x0028,0x0011].value
        else:
            return None

    @property
    def manufacturer(self):
        if (0x0008,0x0070) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0070].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0070].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0070].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0070].value]
            else:
                return self._pydicom_obj[0x0008,0x0070].value
        else:
            return None

    @property
    def patient_support_angle(self):
        if (0x300a,0x0122) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x0122].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x0122].value
            elif (isinstance(self._pydicom_obj[0x300a,0x0122].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300a,0x0122].value]
            else:
                return float(self._pydicom_obj[0x300a,0x0122].value)
        else:
            return None

    @property
    def primary_dosimeter_unit(self):
        if (0x300a,0x00b3) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x00b3].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x00b3].value
            elif (isinstance(self._pydicom_obj[0x300a,0x00b3].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x300a,0x00b3].value]
            else:
                return self._pydicom_obj[0x300a,0x00b3].value
        else:
            return None

    @property
    def series_instance_uid(self):
        if (0x0020,0x000e) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0020,0x000e].value, BaseTag)):
                return self._pydicom_obj[0x0020,0x000e].value
            elif (isinstance(self._pydicom_obj[0x0020,0x000e].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0020,0x000e].value]
            else:
                return self._pydicom_obj[0x0020,0x000e].value
        else:
            return None

    @property
    def image_plane_pixel_spacing(self):
        if (0x3002,0x0011) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3002,0x0011].value, BaseTag)):
                return self._pydicom_obj[0x3002,0x0011].value
            elif (isinstance(self._pydicom_obj[0x3002,0x0011].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x3002,0x0011].value]
            else:
                return float(self._pydicom_obj[0x3002,0x0011].value)
        else:
            return None

    @property
    def content_date(self):
        if (0x0008,0x0023) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0023].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0023].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0023].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0023].value]
            else:
                return self._pydicom_obj[0x0008,0x0023].value
        else:
            return None

    @property
    def conversion_type(self):
        if (0x0008,0x0064) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0064].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0064].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0064].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0064].value]
            else:
                return self._pydicom_obj[0x0008,0x0064].value
        else:
            return None

    @property
    def table_top_vertical_position(self):
        if (0x300a,0x0128) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x0128].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x0128].value
            elif (isinstance(self._pydicom_obj[0x300a,0x0128].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300a,0x0128].value]
            else:
                return float(self._pydicom_obj[0x300a,0x0128].value)
        else:
            return None

    @property
    def rt_image_orientation(self):
        if (0x3002,0x0010) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3002,0x0010].value, BaseTag)):
                return self._pydicom_obj[0x3002,0x0010].value
            elif (isinstance(self._pydicom_obj[0x3002,0x0010].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x3002,0x0010].value]
            else:
                return float(self._pydicom_obj[0x3002,0x0010].value)
        else:
            return None

    @property
    def radiation_machine_sad(self):
        if (0x3002,0x0022) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3002,0x0022].value, BaseTag)):
                return self._pydicom_obj[0x3002,0x0022].value
            elif (isinstance(self._pydicom_obj[0x3002,0x0022].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x3002,0x0022].value]
            else:
                return float(self._pydicom_obj[0x3002,0x0022].value)
        else:
            return None

    @property
    def bits_allocated(self):
        if (0x0028,0x0100) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x0100].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x0100].value
            elif (isinstance(self._pydicom_obj[0x0028,0x0100].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0028,0x0100].value]
            else:
                return self._pydicom_obj[0x0028,0x0100].value
        else:
            return None

    @property
    def samples_per_pixel(self):
        if (0x0028,0x0002) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x0002].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x0002].value
            elif (isinstance(self._pydicom_obj[0x0028,0x0002].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0028,0x0002].value]
            else:
                return self._pydicom_obj[0x0028,0x0002].value
        else:
            return None

    @property
    def table_top_longitudinal_position(self):
        if (0x300a,0x0129) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x0129].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x0129].value
            elif (isinstance(self._pydicom_obj[0x300a,0x0129].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300a,0x0129].value]
            else:
                return float(self._pydicom_obj[0x300a,0x0129].value)
        else:
            return None

    @property
    def gantry_angle(self):
        if (0x300a,0x011e) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x011e].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x011e].value
            elif (isinstance(self._pydicom_obj[0x300a,0x011e].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300a,0x011e].value]
            else:
                return float(self._pydicom_obj[0x300a,0x011e].value)
        else:
            return None

    @property
    def pixel_intensity_relationship_sign(self):
        if (0x0028,0x1041) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x1041].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x1041].value
            elif (isinstance(self._pydicom_obj[0x0028,0x1041].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0028,0x1041].value]
            else:
                return self._pydicom_obj[0x0028,0x1041].value
        else:
            return None

    @property
    def study_id(self):
        if (0x0020,0x0010) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0020,0x0010].value, BaseTag)):
                return self._pydicom_obj[0x0020,0x0010].value
            elif (isinstance(self._pydicom_obj[0x0020,0x0010].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0020,0x0010].value]
            else:
                return self._pydicom_obj[0x0020,0x0010].value
        else:
            return None

    @property
    def instance_creation_date(self):
        if (0x0008,0x0012) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0012].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0012].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0012].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0012].value]
            else:
                return self._pydicom_obj[0x0008,0x0012].value
        else:
            return None

    @property
    def rescale_intercept(self):
        if (0x0028,0x1052) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x1052].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x1052].value
            elif (isinstance(self._pydicom_obj[0x0028,0x1052].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x0028,0x1052].value]
            else:
                return float(self._pydicom_obj[0x0028,0x1052].value)
        else:
            return None

    @property
    def instance_creation_time(self):
        if (0x0008,0x0013) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0013].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0013].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0013].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0013].value]
            else:
                return self._pydicom_obj[0x0008,0x0013].value
        else:
            return None

    @property
    def rt_image_plane(self):
        if (0x3002,0x000c) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3002,0x000c].value, BaseTag)):
                return self._pydicom_obj[0x3002,0x000c].value
            elif (isinstance(self._pydicom_obj[0x3002,0x000c].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x3002,0x000c].value]
            else:
                return self._pydicom_obj[0x3002,0x000c].value
        else:
            return None

    @property
    def window_center(self):
        if (0x0028,0x1050) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x1050].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x1050].value
            elif (isinstance(self._pydicom_obj[0x0028,0x1050].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x0028,0x1050].value]
            else:
                return float(self._pydicom_obj[0x0028,0x1050].value)
        else:
            return None

    @property
    def referenced_fraction_group_number(self):
        if (0x300c,0x0022) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300c,0x0022].value, BaseTag)):
                return self._pydicom_obj[0x300c,0x0022].value
            elif (isinstance(self._pydicom_obj[0x300c,0x0022].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x300c,0x0022].value]
            else:
                return self._pydicom_obj[0x300c,0x0022].value
        else:
            return None

    @property
    def rows(self):
        if (0x0028,0x0010) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x0010].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x0010].value
            elif (isinstance(self._pydicom_obj[0x0028,0x0010].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0028,0x0010].value]
            else:
                return self._pydicom_obj[0x0028,0x0010].value
        else:
            return None

    @property
    def patients_name(self):
        if (0x0010,0x0010) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0010,0x0010].value, BaseTag)):
                return self._pydicom_obj[0x0010,0x0010].value
            elif (isinstance(self._pydicom_obj[0x0010,0x0010].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0010,0x0010].value]
            else:
                return self._pydicom_obj[0x0010,0x0010].value
        else:
            return None

    @property
    def series_number(self):
        if (0x0020,0x0011) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0020,0x0011].value, BaseTag)):
                return self._pydicom_obj[0x0020,0x0011].value
            elif (isinstance(self._pydicom_obj[0x0020,0x0011].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0020,0x0011].value]
            else:
                return self._pydicom_obj[0x0020,0x0011].value
        else:
            return None

    @property
    def manufacturers_model_name(self):
        if (0x0008,0x1090) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x1090].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x1090].value
            elif (isinstance(self._pydicom_obj[0x0008,0x1090].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x1090].value]
            else:
                return self._pydicom_obj[0x0008,0x1090].value
        else:
            return None

    @property
    def acquisition_number(self):
        if (0x0020,0x0012) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0020,0x0012].value, BaseTag)):
                return self._pydicom_obj[0x0020,0x0012].value
            elif (isinstance(self._pydicom_obj[0x0020,0x0012].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0020,0x0012].value]
            else:
                return self._pydicom_obj[0x0020,0x0012].value
        else:
            return None

    @property
    def accession_number(self):
        if (0x0008,0x0050) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x0050].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x0050].value
            elif (isinstance(self._pydicom_obj[0x0008,0x0050].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x0050].value]
            else:
                return self._pydicom_obj[0x0008,0x0050].value
        else:
            return None

    @property
    def xray_image_receptor_translation(self):
        if (0x3002,0x000d) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3002,0x000d].value, BaseTag)):
                return self._pydicom_obj[0x3002,0x000d].value
            elif (isinstance(self._pydicom_obj[0x3002,0x000d].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x3002,0x000d].value]
            else:
                return float(self._pydicom_obj[0x3002,0x000d].value)
        else:
            return None

    @property
    def reported_values_origin(self):
        if (0x3002,0x000a) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3002,0x000a].value, BaseTag)):
                return self._pydicom_obj[0x3002,0x000a].value
            elif (isinstance(self._pydicom_obj[0x3002,0x000a].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x3002,0x000a].value]
            else:
                return self._pydicom_obj[0x3002,0x000a].value
        else:
            return None

    @property
    def bits_stored(self):
        if (0x0028,0x0101) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x0101].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x0101].value
            elif (isinstance(self._pydicom_obj[0x0028,0x0101].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0028,0x0101].value]
            else:
                return self._pydicom_obj[0x0028,0x0101].value
        else:
            return None

    @property
    def fraction_number(self):
        if (0x3002,0x0029) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3002,0x0029].value, BaseTag)):
                return self._pydicom_obj[0x3002,0x0029].value
            elif (isinstance(self._pydicom_obj[0x3002,0x0029].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x3002,0x0029].value]
            else:
                return self._pydicom_obj[0x3002,0x0029].value
        else:
            return None

    @property
    def rt_image_description(self):
        if (0x3002,0x0004) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3002,0x0004].value, BaseTag)):
                return self._pydicom_obj[0x3002,0x0004].value
            elif (isinstance(self._pydicom_obj[0x3002,0x0004].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x3002,0x0004].value]
            else:
                return self._pydicom_obj[0x3002,0x0004].value
        else:
            return None

    @property
    def instance_number(self):
        if (0x0020,0x0013) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0020,0x0013].value, BaseTag)):
                return self._pydicom_obj[0x0020,0x0013].value
            elif (isinstance(self._pydicom_obj[0x0020,0x0013].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0020,0x0013].value]
            else:
                return self._pydicom_obj[0x0020,0x0013].value
        else:
            return None

    @property
    def patient_orientation(self):
        if (0x0020,0x0020) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0020,0x0020].value, BaseTag)):
                return self._pydicom_obj[0x0020,0x0020].value
            elif (isinstance(self._pydicom_obj[0x0020,0x0020].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0020,0x0020].value]
            else:
                return self._pydicom_obj[0x0020,0x0020].value
        else:
            return None

    @property
    def xml_stream(self):
        if (0x3253,0x1000) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3253,0x1000].value, BaseTag)):
                return self._pydicom_obj[0x3253,0x1000].value
            elif (isinstance(self._pydicom_obj[0x3253,0x1000].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x3253,0x1000].value]
            else:
                return self._pydicom_obj[0x3253,0x1000].value
        else:
            return None

    @property
    def data_length_of_the_xml_stream(self):
        if (0x3253,0x1001) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3253,0x1001].value, BaseTag)):
                return self._pydicom_obj[0x3253,0x1001].value
            elif (isinstance(self._pydicom_obj[0x3253,0x1001].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x3253,0x1001].value]
            else:
                return self._pydicom_obj[0x3253,0x1001].value
        else:
            return None

    @property
    def patient_position(self):
        if (0x0018,0x5100) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0018,0x5100].value, BaseTag)):
                return self._pydicom_obj[0x0018,0x5100].value
            elif (isinstance(self._pydicom_obj[0x0018,0x5100].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0018,0x5100].value]
            else:
                return self._pydicom_obj[0x0018,0x5100].value
        else:
            return None

    @property
    def type_of_data(self):
        if (0x5000,0x0020) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x5000,0x0020].value, BaseTag)):
                return self._pydicom_obj[0x5000,0x0020].value
            elif (isinstance(self._pydicom_obj[0x5000,0x0020].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x5000,0x0020].value]
            else:
                return self._pydicom_obj[0x5000,0x0020].value
        else:
            return None

    @property
    def planar_configuration(self):
        if (0x0028,0x0006) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x0006].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x0006].value
            elif (isinstance(self._pydicom_obj[0x0028,0x0006].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0028,0x0006].value]
            else:
                return self._pydicom_obj[0x0028,0x0006].value
        else:
            return None

    @property
    def curve_dimensions(self):
        if (0x5000,0x0005) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x5000,0x0005].value, BaseTag)):
                return self._pydicom_obj[0x5000,0x0005].value
            elif (isinstance(self._pydicom_obj[0x5000,0x0005].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x5000,0x0005].value]
            else:
                return self._pydicom_obj[0x5000,0x0005].value
        else:
            return None

    @property
    def patients_birth_date(self):
        if (0x0010,0x0030) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0010,0x0030].value, BaseTag)):
                return self._pydicom_obj[0x0010,0x0030].value
            elif (isinstance(self._pydicom_obj[0x0010,0x0030].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0010,0x0030].value]
            else:
                return self._pydicom_obj[0x0010,0x0030].value
        else:
            return None

    @property
    def radiation_machine_name(self):
        if (0x3002,0x0020) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3002,0x0020].value, BaseTag)):
                return self._pydicom_obj[0x3002,0x0020].value
            elif (isinstance(self._pydicom_obj[0x3002,0x0020].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x3002,0x0020].value]
            else:
                return self._pydicom_obj[0x3002,0x0020].value
        else:
            return None

    @property
    def axis_labels(self):
        if (0x5000,0x0040) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x5000,0x0040].value, BaseTag)):
                return self._pydicom_obj[0x5000,0x0040].value
            elif (isinstance(self._pydicom_obj[0x5000,0x0040].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x5000,0x0040].value]
            else:
                return self._pydicom_obj[0x5000,0x0040].value
        else:
            return None

    @property
    def photometric_interpretation(self):
        if (0x0028,0x0004) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x0004].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x0004].value
            elif (isinstance(self._pydicom_obj[0x0028,0x0004].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0028,0x0004].value]
            else:
                return self._pydicom_obj[0x0028,0x0004].value
        else:
            return None

    @property
    def rescale_type(self):
        if (0x0028,0x1054) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x1054].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x1054].value
            elif (isinstance(self._pydicom_obj[0x0028,0x1054].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0028,0x1054].value]
            else:
                return self._pydicom_obj[0x0028,0x1054].value
        else:
            return None

    @property
    def xray_image_receptor_angle(self):
        if (0x3002,0x000e) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3002,0x000e].value, BaseTag)):
                return self._pydicom_obj[0x3002,0x000e].value
            elif (isinstance(self._pydicom_obj[0x3002,0x000e].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x3002,0x000e].value]
            else:
                return float(self._pydicom_obj[0x3002,0x000e].value)
        else:
            return None

    @property
    def exposure_sequence(self):
        val = self._pydicom_obj[0x3002,0x0030].value
        if val=='None' or val is None:
            return None
        else:
            return [MVEPIDImageExposureAutoDicom(s) for s in val]
    @property
    def referenced_rt_plan_sequence(self):
        val = self._pydicom_obj[0x300c,0x0002].value
        if val=='None' or val is None:
            return None
        else:
            return [MVEPIDImageReferencedRTPlanAutoDicom(s) for s in val]


    def __str__(self):
        to_ret_str = ''
        to_ret_str += 'sop_instance_uid: %s\n'%(self.sop_instance_uid)
        to_ret_str += 'isocenter_position: %s\n'%(self.isocenter_position)
        to_ret_str += 'study_time: %s\n'%(self.study_time)
        to_ret_str += 'data_value_representation: %s\n'%(self.data_value_representation)
        to_ret_str += 'rt_image_position: %s\n'%(self.rt_image_position)
        to_ret_str += 'modality: %s\n'%(self.modality)
        to_ret_str += 'patient_id: %s\n'%(self.patient_id)
        to_ret_str += 'rescale_slope: %s\n'%(self.rescale_slope)
        to_ret_str += 'referenced_beam_number: %s\n'%(self.referenced_beam_number)
        to_ret_str += 'table_top_lateral_position: %s\n'%(self.table_top_lateral_position)
        to_ret_str += 'irradiation_event_uid: %s\n'%(self.irradiation_event_uid)
        to_ret_str += 'acquisition_time: %s\n'%(self.acquisition_time)
        to_ret_str += 'frame_of_reference_uid: %s\n'%(self.frame_of_reference_uid)
        to_ret_str += 'beam_limiting_device_angle: %s\n'%(self.beam_limiting_device_angle)
        to_ret_str += 'start_cumulative_meterset_weight: %s\n'%(self.start_cumulative_meterset_weight)
        to_ret_str += 'sop_class_uid: %s\n'%(self.sop_class_uid)
        to_ret_str += 'referring_physicians_name: %s\n'%(self.referring_physicians_name)
        to_ret_str += 'rt_image_label: %s\n'%(self.rt_image_label)
        to_ret_str += 'patients_sex: %s\n'%(self.patients_sex)
        to_ret_str += 'number_of_points: %s\n'%(self.number_of_points)
        to_ret_str += 'pixel_representation: %s\n'%(self.pixel_representation)
        to_ret_str += 'curve_data: %s\n'%(self.curve_data)
        to_ret_str += 'end_cumulative_meterset_weight: %s\n'%(self.end_cumulative_meterset_weight)
        to_ret_str += 'device_serial_number: %s\n'%(self.device_serial_number)
        to_ret_str += 'curve_label: %s\n'%(self.curve_label)
        to_ret_str += 'station_name: %s\n'%(self.station_name)
        to_ret_str += 'position_reference_indicator: %s\n'%(self.position_reference_indicator)
        to_ret_str += 'study_date: %s\n'%(self.study_date)
        to_ret_str += 'operators_name: %s\n'%(self.operators_name)
        to_ret_str += 'extended_interface_format_tag: %s\n'%(self.extended_interface_format_tag)
        to_ret_str += 'study_instance_uid: %s\n'%(self.study_instance_uid)
        to_ret_str += 'acquisition_date: %s\n'%(self.acquisition_date)
        to_ret_str += 'specific_character_set: %s\n'%(self.specific_character_set)
        to_ret_str += 'content_time: %s\n'%(self.content_time)
        to_ret_str += 'window_width: %s\n'%(self.window_width)
        to_ret_str += 'software_versions: %s\n'%(self.software_versions)
        to_ret_str += 'axis_units: %s\n'%(self.axis_units)
        to_ret_str += 'high_bit: %s\n'%(self.high_bit)
        to_ret_str += 'rt_image_sid: %s\n'%(self.rt_image_sid)
        to_ret_str += 'image_type: %s\n'%(self.image_type)
        to_ret_str += 'pixel_intensity_relationship: %s\n'%(self.pixel_intensity_relationship)
        to_ret_str += 'columns: %s\n'%(self.columns)
        to_ret_str += 'manufacturer: %s\n'%(self.manufacturer)
        to_ret_str += 'patient_support_angle: %s\n'%(self.patient_support_angle)
        to_ret_str += 'primary_dosimeter_unit: %s\n'%(self.primary_dosimeter_unit)
        to_ret_str += 'series_instance_uid: %s\n'%(self.series_instance_uid)
        to_ret_str += 'image_plane_pixel_spacing: %s\n'%(self.image_plane_pixel_spacing)
        to_ret_str += 'content_date: %s\n'%(self.content_date)
        to_ret_str += 'conversion_type: %s\n'%(self.conversion_type)
        to_ret_str += 'table_top_vertical_position: %s\n'%(self.table_top_vertical_position)
        to_ret_str += 'rt_image_orientation: %s\n'%(self.rt_image_orientation)
        to_ret_str += 'radiation_machine_sad: %s\n'%(self.radiation_machine_sad)
        to_ret_str += 'bits_allocated: %s\n'%(self.bits_allocated)
        to_ret_str += 'samples_per_pixel: %s\n'%(self.samples_per_pixel)
        to_ret_str += 'table_top_longitudinal_position: %s\n'%(self.table_top_longitudinal_position)
        to_ret_str += 'gantry_angle: %s\n'%(self.gantry_angle)
        to_ret_str += 'pixel_intensity_relationship_sign: %s\n'%(self.pixel_intensity_relationship_sign)
        to_ret_str += 'study_id: %s\n'%(self.study_id)
        to_ret_str += 'instance_creation_date: %s\n'%(self.instance_creation_date)
        to_ret_str += 'rescale_intercept: %s\n'%(self.rescale_intercept)
        to_ret_str += 'instance_creation_time: %s\n'%(self.instance_creation_time)
        to_ret_str += 'rt_image_plane: %s\n'%(self.rt_image_plane)
        to_ret_str += 'window_center: %s\n'%(self.window_center)
        to_ret_str += 'referenced_fraction_group_number: %s\n'%(self.referenced_fraction_group_number)
        to_ret_str += 'rows: %s\n'%(self.rows)
        to_ret_str += 'patients_name: %s\n'%(self.patients_name)
        to_ret_str += 'series_number: %s\n'%(self.series_number)
        to_ret_str += 'manufacturers_model_name: %s\n'%(self.manufacturers_model_name)
        to_ret_str += 'acquisition_number: %s\n'%(self.acquisition_number)
        to_ret_str += 'accession_number: %s\n'%(self.accession_number)
        to_ret_str += 'xray_image_receptor_translation: %s\n'%(self.xray_image_receptor_translation)
        to_ret_str += 'reported_values_origin: %s\n'%(self.reported_values_origin)
        to_ret_str += 'bits_stored: %s\n'%(self.bits_stored)
        to_ret_str += 'fraction_number: %s\n'%(self.fraction_number)
        to_ret_str += 'rt_image_description: %s\n'%(self.rt_image_description)
        to_ret_str += 'instance_number: %s\n'%(self.instance_number)
        to_ret_str += 'patient_orientation: %s\n'%(self.patient_orientation)
        to_ret_str += 'xml_stream: %s\n'%(self.xml_stream)
        to_ret_str += 'data_length_of_the_xml_stream: %s\n'%(self.data_length_of_the_xml_stream)
        to_ret_str += 'patient_position: %s\n'%(self.patient_position)
        to_ret_str += 'type_of_data: %s\n'%(self.type_of_data)
        to_ret_str += 'planar_configuration: %s\n'%(self.planar_configuration)
        to_ret_str += 'curve_dimensions: %s\n'%(self.curve_dimensions)
        to_ret_str += 'patients_birth_date: %s\n'%(self.patients_birth_date)
        to_ret_str += 'radiation_machine_name: %s\n'%(self.radiation_machine_name)
        to_ret_str += 'axis_labels: %s\n'%(self.axis_labels)
        to_ret_str += 'photometric_interpretation: %s\n'%(self.photometric_interpretation)
        to_ret_str += 'rescale_type: %s\n'%(self.rescale_type)
        to_ret_str += 'xray_image_receptor_angle: %s\n'%(self.xray_image_receptor_angle)
        to_ret_str += 'exposure_sequence: %s\n'%(self.exposure_sequence)
        to_ret_str += 'referenced_rt_plan_sequence: %s\n'%(self.referenced_rt_plan_sequence)
        return to_ret_str

    @classmethod
    def from_dicom_file(cls, path, stop_before_pixels=False):
        """
         :rtype: MVEPIDImageAutoDicom
        """
        try:
            import dicom as pydicom
        except ImportError:
            import pydicom
        dcm_hand = pydicom.read_file(path,stop_before_pixels=stop_before_pixels)
        to_ret = cls(dcm_hand, path)
        return to_ret

from pydicom.tag import BaseTag
from pydicom.multival import MultiValue
from FilmLab.dicom import PythonFileAutoDicom, PythonAutoDicom, PythonImageFileAutoDicom 

class MVEPIDImageExposureAutoDicom(PythonAutoDicom):
    description = "AutoDicom inner class of MVEPIDImageAutoDicom"
    @property
    def number_of_blocks(self):
        if (0x300a,0x00f0) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x00f0].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x00f0].value
            elif (isinstance(self._pydicom_obj[0x300a,0x00f0].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x300a,0x00f0].value]
            else:
                return self._pydicom_obj[0x300a,0x00f0].value
        else:
            return None

    @property
    def table_top_lateral_position(self):
        if (0x300a,0x012a) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x012a].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x012a].value
            elif (isinstance(self._pydicom_obj[0x300a,0x012a].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300a,0x012a].value]
            else:
                return float(self._pydicom_obj[0x300a,0x012a].value)
        else:
            return None

    @property
    def table_top_vertical_position(self):
        if (0x300a,0x0128) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x0128].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x0128].value
            elif (isinstance(self._pydicom_obj[0x300a,0x0128].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300a,0x0128].value]
            else:
                return float(self._pydicom_obj[0x300a,0x0128].value)
        else:
            return None

    @property
    def referenced_frame_number(self):
        if (0x0008,0x1160) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x1160].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x1160].value
            elif (isinstance(self._pydicom_obj[0x0008,0x1160].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x1160].value]
            else:
                return self._pydicom_obj[0x0008,0x1160].value
        else:
            return None

    @property
    def patient_support_angle(self):
        if (0x300a,0x0122) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x0122].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x0122].value
            elif (isinstance(self._pydicom_obj[0x300a,0x0122].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300a,0x0122].value]
            else:
                return float(self._pydicom_obj[0x300a,0x0122].value)
        else:
            return None

    @property
    def kvp(self):
        if (0x0018,0x0060) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0018,0x0060].value, BaseTag)):
                return self._pydicom_obj[0x0018,0x0060].value
            elif (isinstance(self._pydicom_obj[0x0018,0x0060].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x0018,0x0060].value]
            else:
                return float(self._pydicom_obj[0x0018,0x0060].value)
        else:
            return None

    @property
    def beam_limiting_device_angle(self):
        if (0x300a,0x0120) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x0120].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x0120].value
            elif (isinstance(self._pydicom_obj[0x300a,0x0120].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300a,0x0120].value]
            else:
                return float(self._pydicom_obj[0x300a,0x0120].value)
        else:
            return None

    @property
    def table_top_longitudinal_position(self):
        if (0x300a,0x0129) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x0129].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x0129].value
            elif (isinstance(self._pydicom_obj[0x300a,0x0129].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300a,0x0129].value]
            else:
                return float(self._pydicom_obj[0x300a,0x0129].value)
        else:
            return None

    @property
    def gantry_angle(self):
        if (0x300a,0x011e) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x011e].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x011e].value
            elif (isinstance(self._pydicom_obj[0x300a,0x011e].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300a,0x011e].value]
            else:
                return float(self._pydicom_obj[0x300a,0x011e].value)
        else:
            return None

    @property
    def meterset_exposure(self):
        if (0x3002,0x0032) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3002,0x0032].value, BaseTag)):
                return self._pydicom_obj[0x3002,0x0032].value
            elif (isinstance(self._pydicom_obj[0x3002,0x0032].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x3002,0x0032].value]
            else:
                return float(self._pydicom_obj[0x3002,0x0032].value)
        else:
            return None

    @property
    def exposure_time(self):
        if (0x0018,0x1150) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0018,0x1150].value, BaseTag)):
                return self._pydicom_obj[0x0018,0x1150].value
            elif (isinstance(self._pydicom_obj[0x0018,0x1150].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0018,0x1150].value]
            else:
                return self._pydicom_obj[0x0018,0x1150].value
        else:
            return None

    @property
    def beam_limiting_device_sequence(self):
        val = self._pydicom_obj[0x300a,0x00b6].value
        if val=='None' or val is None:
            return None
        else:
            return [MVEPIDImageBeamLimitingDeviceAutoDicom(s) for s in val]


    def __str__(self):
        to_ret_str = ''
        to_ret_str += 'number_of_blocks: %s\n'%(self.number_of_blocks)
        to_ret_str += 'table_top_lateral_position: %s\n'%(self.table_top_lateral_position)
        to_ret_str += 'table_top_vertical_position: %s\n'%(self.table_top_vertical_position)
        to_ret_str += 'referenced_frame_number: %s\n'%(self.referenced_frame_number)
        to_ret_str += 'patient_support_angle: %s\n'%(self.patient_support_angle)
        to_ret_str += 'kvp: %s\n'%(self.kvp)
        to_ret_str += 'beam_limiting_device_angle: %s\n'%(self.beam_limiting_device_angle)
        to_ret_str += 'table_top_longitudinal_position: %s\n'%(self.table_top_longitudinal_position)
        to_ret_str += 'gantry_angle: %s\n'%(self.gantry_angle)
        to_ret_str += 'meterset_exposure: %s\n'%(self.meterset_exposure)
        to_ret_str += 'exposure_time: %s\n'%(self.exposure_time)
        to_ret_str += 'beam_limiting_device_sequence: %s\n'%(self.beam_limiting_device_sequence)
        return to_ret_str
from pydicom.tag import BaseTag
from pydicom.multival import MultiValue
from FilmLab.dicom import PythonFileAutoDicom, PythonAutoDicom, PythonImageFileAutoDicom 

class MVEPIDImageBeamLimitingDeviceAutoDicom(PythonAutoDicom):
    description = "AutoDicom inner class of MVEPIDImageExposureAutoDicom"
    @property
    def number_of_leafjaw_pairs(self):
        if (0x300a,0x00bc) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x00bc].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x00bc].value
            elif (isinstance(self._pydicom_obj[0x300a,0x00bc].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x300a,0x00bc].value]
            else:
                return self._pydicom_obj[0x300a,0x00bc].value
        else:
            return None

    @property
    def leafjaw_positions(self):
        if (0x300a,0x011c) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x011c].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x011c].value
            elif (isinstance(self._pydicom_obj[0x300a,0x011c].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x300a,0x011c].value]
            else:
                return float(self._pydicom_obj[0x300a,0x011c].value)
        else:
            return None

    @property
    def rt_beam_limiting_device_type(self):
        if (0x300a,0x00b8) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x300a,0x00b8].value, BaseTag)):
                return self._pydicom_obj[0x300a,0x00b8].value
            elif (isinstance(self._pydicom_obj[0x300a,0x00b8].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x300a,0x00b8].value]
            else:
                return self._pydicom_obj[0x300a,0x00b8].value
        else:
            return None



    def __str__(self):
        to_ret_str = ''
        to_ret_str += 'number_of_leafjaw_pairs: %s\n'%(self.number_of_leafjaw_pairs)
        to_ret_str += 'leafjaw_positions: %s\n'%(self.leafjaw_positions)
        to_ret_str += 'rt_beam_limiting_device_type: %s\n'%(self.rt_beam_limiting_device_type)
        return to_ret_str


from pydicom.tag import BaseTag
from pydicom.multival import MultiValue
from FilmLab.dicom import PythonFileAutoDicom, PythonAutoDicom, PythonImageFileAutoDicom 

class MVEPIDImageReferencedRTPlanAutoDicom(PythonAutoDicom):
    description = "AutoDicom inner class of MVEPIDImageAutoDicom"
    @property
    def referenced_sop_class_uid(self):
        if (0x0008,0x1150) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x1150].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x1150].value
            elif (isinstance(self._pydicom_obj[0x0008,0x1150].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x1150].value]
            else:
                return self._pydicom_obj[0x0008,0x1150].value
        else:
            return None

    @property
    def referenced_sop_instance_uid(self):
        if (0x0008,0x1155) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x1155].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x1155].value
            elif (isinstance(self._pydicom_obj[0x0008,0x1155].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x1155].value]
            else:
                return self._pydicom_obj[0x0008,0x1155].value
        else:
            return None



    def __str__(self):
        to_ret_str = ''
        to_ret_str += 'referenced_sop_class_uid: %s\n'%(self.referenced_sop_class_uid)
        to_ret_str += 'referenced_sop_instance_uid: %s\n'%(self.referenced_sop_instance_uid)
        return to_ret_str

