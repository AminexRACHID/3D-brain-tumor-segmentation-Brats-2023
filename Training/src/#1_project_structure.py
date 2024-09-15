import os

structure = {
    'data': {
        'training': {
            'images': {},
            'masks': {},
            'SAD_Unet_raw': {
                'Dataset001_BRATS': {
                    'dataset.json': None,
                    'imagesTr': {},
                    'labelsTr': {}
                }
            },
            'SAD_Unet_preprocessed': {},
            'SAD_Unet_results': {}
        },
        'test': {
            'images': {},
            'masks': {},
            'SAD_Unet_raw': {
                'Dataset001_BRATS': {
                    'imagesTs': {},
                    'labelsTs': {}
                }
            }
        }
    }
}
def create_structure(base_path, structure):
    for dir_name, sub_structure in structure.items():
        path = os.path.join(base_path, dir_name)
        os.makedirs(path, exist_ok=True)
        
        if sub_structure:
            create_structure(path, sub_structure)

create_structure('..', structure)
