from pydicom.tag import BaseTag
from pydicom.multival import MultiValue
from FilmLab.dicom import PythonFileAutoDicom, PythonAutoDicom, PythonImageFileAutoDicom 

class VICDoseAutoDicom(PythonImageFileAutoDicom):
    description = "VIC dose storage format"
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
    def tissue_heterogeneity_correction(self):
        if (0x3004,0x0014) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3004,0x0014].value, BaseTag)):
                return self._pydicom_obj[0x3004,0x0014].value
            elif (isinstance(self._pydicom_obj[0x3004,0x0014].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x3004,0x0014].value]
            else:
                return self._pydicom_obj[0x3004,0x0014].value
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
    def dose_summation_type(self):
        if (0x3004,0x000a) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3004,0x000a].value, BaseTag)):
                return self._pydicom_obj[0x3004,0x000a].value
            elif (isinstance(self._pydicom_obj[0x3004,0x000a].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x3004,0x000a].value]
            else:
                return self._pydicom_obj[0x3004,0x000a].value
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
    def slice_thickness(self):
        if (0x0018,0x0050) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0018,0x0050].value, BaseTag)):
                return self._pydicom_obj[0x0018,0x0050].value
            elif (isinstance(self._pydicom_obj[0x0018,0x0050].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x0018,0x0050].value]
            else:
                return float(self._pydicom_obj[0x0018,0x0050].value)
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
    def dose_units(self):
        if (0x3004,0x0002) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3004,0x0002].value, BaseTag)):
                return self._pydicom_obj[0x3004,0x0002].value
            elif (isinstance(self._pydicom_obj[0x3004,0x0002].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x3004,0x0002].value]
            else:
                return self._pydicom_obj[0x3004,0x0002].value
        else:
            return None

    @property
    def number_of_frames(self):
        if (0x0028,0x0008) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x0008].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x0008].value
            elif (isinstance(self._pydicom_obj[0x0028,0x0008].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0028,0x0008].value]
            else:
                return self._pydicom_obj[0x0028,0x0008].value
        else:
            return None

    @property
    def image_orientation_patient(self):
        if (0x0020,0x0037) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0020,0x0037].value, BaseTag)):
                return self._pydicom_obj[0x0020,0x0037].value
            elif (isinstance(self._pydicom_obj[0x0020,0x0037].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x0020,0x0037].value]
            else:
                return float(self._pydicom_obj[0x0020,0x0037].value)
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
    def grid_frame_offset_vector(self):
        if (0x3004,0x000c) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3004,0x000c].value, BaseTag)):
                return self._pydicom_obj[0x3004,0x000c].value
            elif (isinstance(self._pydicom_obj[0x3004,0x000c].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x3004,0x000c].value]
            else:
                return float(self._pydicom_obj[0x3004,0x000c].value)
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
    def dose_grid_scaling(self):
        if (0x3004,0x000e) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3004,0x000e].value, BaseTag)):
                return self._pydicom_obj[0x3004,0x000e].value
            elif (isinstance(self._pydicom_obj[0x3004,0x000e].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x3004,0x000e].value]
            else:
                return float(self._pydicom_obj[0x3004,0x000e].value)
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
    def image_position_patient(self):
        if (0x0020,0x0032) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0020,0x0032].value, BaseTag)):
                return self._pydicom_obj[0x0020,0x0032].value
            elif (isinstance(self._pydicom_obj[0x0020,0x0032].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x0020,0x0032].value]
            else:
                return float(self._pydicom_obj[0x0020,0x0032].value)
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
    def study_description(self):
        if (0x0008,0x1030) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x1030].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x1030].value
            elif (isinstance(self._pydicom_obj[0x0008,0x1030].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x1030].value]
            else:
                return self._pydicom_obj[0x0008,0x1030].value
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
    def series_description(self):
        if (0x0008,0x103e) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0008,0x103e].value, BaseTag)):
                return self._pydicom_obj[0x0008,0x103e].value
            elif (isinstance(self._pydicom_obj[0x0008,0x103e].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0008,0x103e].value]
            else:
                return self._pydicom_obj[0x0008,0x103e].value
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
    def pixel_spacing(self):
        if (0x0028,0x0030) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x0030].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x0030].value
            elif (isinstance(self._pydicom_obj[0x0028,0x0030].value, MultiValue)):
                return [float(s) for s in self._pydicom_obj[0x0028,0x0030].value]
            else:
                return float(self._pydicom_obj[0x0028,0x0030].value)
        else:
            return None

    @property
    def frame_increment_pointer(self):
        if (0x0028,0x0009) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x0028,0x0009].value, BaseTag)):
                return self._pydicom_obj[0x0028,0x0009].value
            elif (isinstance(self._pydicom_obj[0x0028,0x0009].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x0028,0x0009].value]
            else:
                return self._pydicom_obj[0x0028,0x0009].value
        else:
            return None

    @property
    def dose_type(self):
        if (0x3004,0x0004) in self._pydicom_obj:
            if (isinstance(self._pydicom_obj[0x3004,0x0004].value, BaseTag)):
                return self._pydicom_obj[0x3004,0x0004].value
            elif (isinstance(self._pydicom_obj[0x3004,0x0004].value, MultiValue)):
                return [s for s in self._pydicom_obj[0x3004,0x0004].value]
            else:
                return self._pydicom_obj[0x3004,0x0004].value
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
    def referenced_rt_plan_sequence(self):
        val = self._pydicom_obj[0x300c,0x0002].value
        if val=='None' or val is None:
            return None
        else:
            return [VICDoseReferencedRTPlanAutoDicom(s) for s in val]


    def __str__(self):
        to_ret_str = ''
        to_ret_str += 'accession_number: %s\n'%(self.accession_number)
        to_ret_str += 'sop_instance_uid: %s\n'%(self.sop_instance_uid)
        to_ret_str += 'device_serial_number: %s\n'%(self.device_serial_number)
        to_ret_str += 'tissue_heterogeneity_correction: %s\n'%(self.tissue_heterogeneity_correction)
        to_ret_str += 'station_name: %s\n'%(self.station_name)
        to_ret_str += 'position_reference_indicator: %s\n'%(self.position_reference_indicator)
        to_ret_str += 'study_time: %s\n'%(self.study_time)
        to_ret_str += 'bits_stored: %s\n'%(self.bits_stored)
        to_ret_str += 'dose_summation_type: %s\n'%(self.dose_summation_type)
        to_ret_str += 'patient_id: %s\n'%(self.patient_id)
        to_ret_str += 'modality: %s\n'%(self.modality)
        to_ret_str += 'study_date: %s\n'%(self.study_date)
        to_ret_str += 'slice_thickness: %s\n'%(self.slice_thickness)
        to_ret_str += 'operators_name: %s\n'%(self.operators_name)
        to_ret_str += 'dose_units: %s\n'%(self.dose_units)
        to_ret_str += 'number_of_frames: %s\n'%(self.number_of_frames)
        to_ret_str += 'image_orientation_patient: %s\n'%(self.image_orientation_patient)
        to_ret_str += 'bits_allocated: %s\n'%(self.bits_allocated)
        to_ret_str += 'study_instance_uid: %s\n'%(self.study_instance_uid)
        to_ret_str += 'grid_frame_offset_vector: %s\n'%(self.grid_frame_offset_vector)
        to_ret_str += 'specific_character_set: %s\n'%(self.specific_character_set)
        to_ret_str += 'samples_per_pixel: %s\n'%(self.samples_per_pixel)
        to_ret_str += 'dose_grid_scaling: %s\n'%(self.dose_grid_scaling)
        to_ret_str += 'study_id: %s\n'%(self.study_id)
        to_ret_str += 'image_position_patient: %s\n'%(self.image_position_patient)
        to_ret_str += 'software_versions: %s\n'%(self.software_versions)
        to_ret_str += 'instance_creation_date: %s\n'%(self.instance_creation_date)
        to_ret_str += 'frame_of_reference_uid: %s\n'%(self.frame_of_reference_uid)
        to_ret_str += 'high_bit: %s\n'%(self.high_bit)
        to_ret_str += 'instance_creation_time: %s\n'%(self.instance_creation_time)
        to_ret_str += 'sop_class_uid: %s\n'%(self.sop_class_uid)
        to_ret_str += 'referring_physicians_name: %s\n'%(self.referring_physicians_name)
        to_ret_str += 'study_description: %s\n'%(self.study_description)
        to_ret_str += 'rows: %s\n'%(self.rows)
        to_ret_str += 'columns: %s\n'%(self.columns)
        to_ret_str += 'patients_sex: %s\n'%(self.patients_sex)
        to_ret_str += 'series_description: %s\n'%(self.series_description)
        to_ret_str += 'patients_birth_date: %s\n'%(self.patients_birth_date)
        to_ret_str += 'manufacturer: %s\n'%(self.manufacturer)
        to_ret_str += 'patients_name: %s\n'%(self.patients_name)
        to_ret_str += 'photometric_interpretation: %s\n'%(self.photometric_interpretation)
        to_ret_str += 'pixel_representation: %s\n'%(self.pixel_representation)
        to_ret_str += 'series_number: %s\n'%(self.series_number)
        to_ret_str += 'manufacturers_model_name: %s\n'%(self.manufacturers_model_name)
        to_ret_str += 'pixel_spacing: %s\n'%(self.pixel_spacing)
        to_ret_str += 'frame_increment_pointer: %s\n'%(self.frame_increment_pointer)
        to_ret_str += 'dose_type: %s\n'%(self.dose_type)
        to_ret_str += 'series_instance_uid: %s\n'%(self.series_instance_uid)
        to_ret_str += 'referenced_rt_plan_sequence: %s\n'%(self.referenced_rt_plan_sequence)
        return to_ret_str

    @classmethod
    def from_dicom_file(cls, path, stop_before_pixels=False):
        """
         :rtype: VICDoseAutoDicom
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

class VICDoseReferencedRTPlanAutoDicom(PythonAutoDicom):
    description = "AutoDicom inner class of VICDoseAutoDicom"
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

