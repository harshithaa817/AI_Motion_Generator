import os
import numpy as np

def create_mock_data():
    base_dir = r'C:\Users\harsh\Desktop\AI_MotionGen\dataset\HumanML3D'
    os.makedirs(os.path.join(base_dir, 'new_joint_vecs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'texts'), exist_ok=True)
    
    # Create Mean and Std if they don't exist
    # Looking at the code, it expects Mean.npy and Std.npy in data_root
    t2m_mean = r'C:\Users\harsh\Desktop\AI_MotionGen\dataset\t2m_mean.npy'
    t2m_std = r'C:\Users\harsh\Desktop\AI_MotionGen\dataset\t2m_std.npy'
    
    import shutil
    if os.path.exists(t2m_mean):
        shutil.copy(t2m_mean, os.path.join(base_dir, 'Mean.npy'))
    if os.path.exists(t2m_std):
        shutil.copy(t2m_std, os.path.join(base_dir, 'Std.npy'))
        
    names = ['000001', '000002', '000003']
    with open(os.path.join(base_dir, 'test.txt'), 'w') as f:
        for name in names:
            f.write(f'{name}\n')
            
    for name in names:
        # Create dummy motion
        np.save(os.path.join(base_dir, 'new_joint_vecs', f'{name}.npy'), np.zeros((100, 263)))
        # Create dummy text
        with open(os.path.join(base_dir, 'texts', f'{name}.txt'), 'w') as f:
            f.write(f'dummy caption {name}#dummy tokens#0.0#0.0\n')
            
    print("Mock data created successfully at", base_dir)

if __name__ == '__main__':
    create_mock_data()
