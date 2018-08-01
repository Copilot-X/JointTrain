import numpy as np

def build_entity2id(instance_triple):
    """
    instance_triple: (e1, e2, r) in str
    """
    entity = dict()
    entry = 0
    for pair in instance_triple:
        e1, e2, _ = pair
        if not e1 in entity:
            entity[e1] = entry
            entry += 1
        if not e2 in entity:
            entity[e2] = entry
            entry += 1
    return entity, entry

def build_train2id(instance_triple, entity2id):
    """
    train2id: (e1_id, e2_id, r) in int
    """
    train2id = list()
    for pair in instance_triple:
        e1, e2, r_id = pair
        e1_id = entity2id[e1]
        e2_id = entity2id[e2]
        train2id.append((e1_id, e2_id, r_id))
    return train2id, len(train2id)

def load_rela2id(filepath):
    rela2id = dict()
    f = open(filepath, 'r')
    total = (int)(f.readline().strip())
    for i in range(total):
        content = f.readline().strip().split()
        rela2id[content[0]] = int(content[1])
    f.close()
    return rela2id, len(rela2id)

if __name__ == '__main__':
    data_path = '../data/'
    export_path = 'data/'
    instance_triple = np.load(data_path + 'train_instance_triple.npy')
    entity2id, ent_num = build_entity2id(instance_triple)
    rela2id, rela_num = load_rela2id(export_path + 'relation2id.txt')
    train2id, train_num = build_train2id(instance_triple, entity2id)
    # save entity2id files
    f = open(export_path + 'entity2id.txt', 'w')
    f.writelines(str(ent_num) + '\n')
    for entity in entity2id:
        f.write('{e}\t{id}\n'.format(e=str(entity), id=entity2id[entity]))
    f.close()
    # save tain2id files
    f = open(export_path + 'train2id.txt', 'w')
    f.write(str(train_num) + '\n')
    for triple in train2id:
        e1_id, e2_id, r_id = triple
        f.write('{e1}\t{e2}\t{r}\n'.format(e1=e1_id, e2=e2_id, r=r_id))
    f.close()
