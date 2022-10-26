import os



# define paths
root = os.pardir
data_root = os.path.join(root, 'data')
paths = dict(
    data=data_root, clean_data=os.path.join(data_root, 'clean_data'), 

    m5_spec=os.path.join(data_root, 'clean_data', 'm5_spec.csv'), 
    mp5_spec=os.path.join(data_root, 'clean_data', 'mp5_spec.csv'),
    mp6_spec=os.path.join(data_root, 'clean_data', 'mp6_spec.csv'),

    label0=os.path.join(data_root, 'clean_data', 'label0.csv'),
    label1=os.path.join(data_root, 'clean_data', 'label1.csv'),
    label2=os.path.join(data_root, 'clean_data', 'label2.csv'), 
    label3=os.path.join(data_root, 'clean_data', 'label3.csv'),

    ckpoint=os.path.join(data_root, 'ckpoint')
)


def main():
    print(paths)



if __name__ == '__main__':
    main()