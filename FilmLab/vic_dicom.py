from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Generic, TypeVar, List

from FilmLab.utils import NDVoxelArray, list_files_by_wildcard
from FilmLab.dicom import PythonAutoDicom, PythonFileAutoDicom

from FilmLab.vic_autodicom.kv_obi_image_autodicom import KVOBIImageAutoDicom
from FilmLab.vic_autodicom.mv_epid_image_autodicom import MVEPIDImageAutoDicom
from FilmLab.vic_autodicom.vic_dose_autodicom import VICDoseAutoDicom
from FilmLab.vic_autodicom.vic_rtplan_autodicom import VICRTPlanAutoDicom
from FilmLab.vic_autodicom.vic_rtplan_autodicom import VICRTPlanAutoDicom


class PairedDicomFile(ABC):

    AUTOLOAD_PAIRED_FILES = True

    def __init__(self, *args, find_paired_files=True, autoload_paired_files=AUTOLOAD_PAIRED_FILES, **kwargs):
        super().__init__(*args, **kwargs)

        assert issubclass(self.PAIRING_TYPE, PythonFileAutoDicom)

        if not hasattr(self, 'paired_files'):
            self.paired_files = {}
            self._paired_dicom = {}

        if find_paired_files:
            print("\tSearching for paired DICOM files...")
            paired_file = self.find_paired_file(self)

            if paired_file is not None:

                if self.PAIRING_TYPE in self.paired_files.keys():
                    raise NotImplementedError("multiple paired files of same type not implemented")

                self.paired_files[self.PAIRING_TYPE] = paired_file

                if autoload_paired_files:
                    self._paired_dicom[self.PAIRING_TYPE] = self.get_paired_dicom(self.PAIRING_TYPE)


    def base_paring_key(self, dicom_handle):
        return self.get_pairing_key(dicom_handle)

    @abstractmethod
    def get_pairing_key(self, dicom_handle) -> str:
        pass

    @property
    @abstractmethod
    def PAIRING_TYPE(self) -> PythonFileAutoDicom:
        pass

    @property
    @abstractmethod
    def PAIRING_WC(self) -> str:
        pass



    @classmethod
    def find_paired_file(cls, file_or_dicom_to_pair, search_dir=None):
        import os

        if isinstance(file_or_dicom_to_pair, str) and os.path.isfile(file_or_dicom_to_pair):
            file_or_dicom_to_pair = cls(file_or_dicom_to_pair)

        if search_dir is None:
            dir_name = os.path.dirname(file_or_dicom_to_pair.file_name)
        else:
            base_dir = search_dir
            assert os.path.isdir(base_dir)

        search_wc = os.path.join(dir_name, cls.PAIRING_WC)
        cand_files = list_files_by_wildcard(search_wc)
        key_to_match = cls.base_paring_key(file_or_dicom_to_pair)

        if issubclass(cls.PAIRING_TYPE, PairedDicomFile):
            id_list = [cls.get_pairing_key(cls.PAIRING_TYPE(s, verbose=False, load_paired_files=False)) for s in cand_files]
        else:
            id_list = [cls.get_pairing_key(cls.PAIRING_TYPE(s, verbose=False)) for s in cand_files]

        matching_id = [ind for ind in range(0,len(id_list)) if id_list[ind] == key_to_match]

        if len(matching_id) == 0:
            print("\t\tFailed to find file matching %s" % cls.PAIRING_WC)
            return None
        elif len(matching_id) == 1:
            print("\t\tFound matching %s file: %s"%(cls.PAIRING_TYPE.__name__,cand_files[matching_id[0]]))
            return cand_files[matching_id[0]]
        else:
            raise RuntimeError('multiple files in dir match dose ID')

    def get_paired_dicom(self, pairing_type: PythonFileAutoDicom, force_reload: bool=False):
        '''
        Load paired dicom object from file
        :param pairing_type: type of paired file inherits PythonFileAutoDicom
        :param force_reload: force reload of dicom if already loaded
        :return: paired DICOM object of pairing_type
        '''
        if pairing_type not in self.paired_files.keys():
            print(self.paired_files.keys())
            raise FileNotFoundError("paired %s file not found"%pairing_type)

        paired_file = self.paired_files[pairing_type]

        if force_reload or not pairing_type in self._paired_dicom.keys():
            if issubclass(pairing_type, PairedDicomFile):
                self._paired_dicom[pairing_type] = pairing_type(paired_file, find_paired_files=False)
            else:
                self._paired_dicom[pairing_type] = pairing_type(paired_file)

        return self._paired_dicom[pairing_type]




class KVOBIImage(KVOBIImageAutoDicom):
    pass

class MVEPIDImage(MVEPIDImageAutoDicom):
    pass

class VICRTPlan(VICRTPlanAutoDicom):

    @property
    def isocenter(self) -> list[float]:

        self.beam_sequence[0].control_point_sequence[0].isocenter_position

        first_iso = self.beam_sequence[0].control_point_sequence[0].isocenter_position

        if not all([c[0].isocenter_position==first_iso
                    for c in [s.control_point_sequence for s in self.beam_sequence]]):
            raise RuntimeError("only single-isocenter plans implemented")
        else:
            return first_iso

class VIDoseStorage(VICDoseAutoDicom):

    @property
    def plan_dicom(self) -> VICRTPlan:
        return self.get_paired_dicom(self.PAIRING_TYPE)

    @property
    def dose_array_cgy(self) -> NDVoxelArray:

        if (self.dose_units == "GY"):
            mult_factor = 100
        else:
            raise NotImplementedError()

        return self._get_ndimage()*self.dose_grid_scaling*mult_factor


    def calculate_dimensions(self):

        row_spacing = self.pixel_spacing[0]
        column_spacing = self.pixel_spacing[1]
        origin = [self.image_position_patient[2],self.image_position_patient[1],self.image_position_patient[0]]

        row_direction_cosine = [self.image_orientation_patient[2], self.image_orientation_patient[1], self.image_orientation_patient[0]]
        column_direction_cosine = [self.image_orientation_patient[5], self.image_orientation_patient[4], self.image_orientation_patient[3]]

        if not all( [a==b for a,b in zip(row_direction_cosine,[0.0,0.0,1.0])]):
            raise NotImplementedError('only x-row, y-col, z-slice dicom implemented!')
        if not all( [a==b for a,b in zip(column_direction_cosine,[0.0,1.0,0.0])]):
            raise NotImplementedError('only x-row, y-col, z-slice dicom implemented!')

        slice_spacing = self.grid_frame_offset_vector[1]

        dz,dy,dx = [slice_spacing, column_spacing, row_spacing]

        origin = [ origin[0]-(dz/2.0), origin[1]-(dy/2.0), origin[2]-(dx/2.0) ]

        return origin, (dz,dy,dx)

class PairedRTPlan(PairedDicomFile):

    @classmethod
    def get_pairing_key(cls, dicom_handle):
        return dicom_handle.sop_instance_uid

    @classmethod
    def base_paring_key(self, dicom_handle):
        return dicom_handle.referenced_rt_plan_sequence[0].referenced_sop_instance_uid

    PAIRING_TYPE = VICRTPlan
    PAIRING_WC = "RP*.dcm"

class PairedDoseFile(PairedDicomFile):

    @classmethod
    def get_pairing_key(cls, dicom_handle):
        return dicom_handle.referenced_rt_plan_sequence[0].referenced_sop_instance_uid

    @classmethod
    def base_paring_key(self, dicom_handle):
        return dicom_handle.sop_instance_uid

    PAIRING_TYPE = VIDoseStorage
    PAIRING_WC = "RD*.dcm"

class FilmLabDoseFile(PairedRTPlan, VIDoseStorage):
    PAIRING_WC = "*RP*.dcm"

