import PagePrepare from '../components/PagePrepare'
import PageJobs from '../components/PageJobViewer'
import PageSettings from '../components/PageSettings'
import PageDocs from '../components/PageDocs'
import PageCommon from '../components/PageCommon'
import {
    DrawerPrepare,
    DrawerRefine,
    DrawerDenoise,
    DrawerPredict,
    DrawerDeconv,
    DrawerMask
} from '../components/Drawer'

import SourceTwoToneIcon from '@mui/icons-material/SourceTwoTone';
import GraphicEqTwoToneIcon from '@mui/icons-material/GraphicEqTwoTone';
import MasksTwoToneIcon from '@mui/icons-material/MasksTwoTone';
import FaceRetouchingNaturalTwoToneIcon from '@mui/icons-material/FaceRetouchingNaturalTwoTone';
import FilterHdrTwoToneIcon from '@mui/icons-material/FilterHdrTwoTone';
import AlbumTwoToneIcon from '@mui/icons-material/AlbumTwoTone';
import BallotTwoToneIcon from '@mui/icons-material/BallotTwoTone';
import DescriptionIcon from '@mui/icons-material/Description';
import SettingsIcon from '@mui/icons-material/Settings';

export const primaryMenuListinOrder = [
    'prepare_star',
    'denoise',
    'deconv',
    'make_mask',
    'refine',
    'predict',
    'jobs_viewer',
    'documents',
    'settings'
]
export const primaryMenuMapping = {
    prepare_star: {
        label: 'Prepare',
        drawer: DrawerPrepare,
        page: PagePrepare,
        icon: SourceTwoToneIcon,
    },
    denoise: {
        label: 'Denoise',
        drawer: DrawerDenoise,
        page: PageCommon,
        icon: GraphicEqTwoToneIcon,
    },
    deconv: {
        label: 'Deconvolve',
        drawer: DrawerDeconv,
        page: PageCommon,
        icon: AlbumTwoToneIcon,
    },
    make_mask: {
        label: 'Create Mask',
        drawer: DrawerMask,
        page: PageCommon,
        icon: MasksTwoToneIcon,
    },
    refine: {
        label: 'Refine',
        drawer: DrawerRefine,
        page: PageCommon,
        icon: FaceRetouchingNaturalTwoToneIcon,
    },
    predict: {
        label: 'Predict',
        drawer: DrawerPredict,
        page: PageCommon,
        icon: FilterHdrTwoToneIcon,
    },
    jobs_viewer: {
        label: 'Jobs Viewer',
        page: PageJobs,
        icon: BallotTwoToneIcon,
    },
    documents: {
        label: 'Documents',
        page: PageDocs,
        icon: DescriptionIcon,
    },
    settings: {
        label: 'Settings',
        page: PageSettings,
        icon: SettingsIcon,
    }
}