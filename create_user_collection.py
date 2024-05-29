#!/usr/bin/env python
from typing import Any, Dict, List
import contextlib
import io
import random
import string

from pymilvus import DataType, MilvusClient
import numpy as np
import pymilvus

EMBEDDING_DIMENSION = 8
NUM_ENTITIES = 300
URI = 'http://localhost:19530'

ROOT_USER = 'root'
ROOT_PASS = 'Milvus'

USERNAME = 'user1'
USERPASS = 'user1_password'
USERROLE = 'user_role'
USER_COLLECTION = 'user_collection'


class FieldName:
    PK = 'pk'
    RANDOM = 'random'
    EMBEDDINGS = 'embeddings'


def remove_user(root_client):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            # .. note:: The typehint is List[Dict], but functionally it is Dict.
            # noinspection PyTypeChecker
            privileges = root_client.describe_role(USERROLE)['privileges']
    except pymilvus.exceptions.MilvusException:
        privileges = []

    # .. note:: describe role is type hinted as List[Dict], but the actual return is Dict.
    for privilege in privileges:
        root_client.revoke_privilege(USERROLE,
                                     privilege['object_type'],
                                     privilege['privilege'],
                                     privilege['object_name'])
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            root_client.revoke_role(USERNAME, USERROLE)
    except pymilvus.exceptions.MilvusException:
        pass

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            root_client.drop_role(USERROLE)
    except pymilvus.exceptions.MilvusException:
        pass

    root_client.drop_user(USERNAME)


def add_user(root_client) -> MilvusClient:
    root_client.create_user(USERNAME, password=USERPASS)

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            root_client.create_role(USERROLE)
    except pymilvus.exceptions.MilvusException as err:
        if 'already exists' not in str(err):
            raise err

    root_client.grant_privilege(role_name=USERROLE,
                                object_type='Global',
                                privilege='All',
                                object_name='*')
    root_client.grant_privilege(role_name=USERROLE,
                                object_type='Global',
                                privilege='CreateCollection',
                                object_name='CreateCollection')
    root_client.grant_privilege(role_name=USERROLE,
                                object_type='Global',
                                privilege='ShowCollections',
                                object_name='ShowCollections')
    root_client.grant_role(USERNAME, USERROLE)

    return MilvusClient(URI, USERNAME, password=USERPASS)


def create_collection(client: MilvusClient):
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_fields=True,
        description="A test collection.",
    )

    schema.add_field(field_name=FieldName.PK,
                     datatype=DataType.VARCHAR, is_primary=True, max_length=100)
    schema.add_field(field_name=FieldName.RANDOM, datatype=DataType.DOUBLE)
    schema.add_field(field_name=FieldName.EMBEDDINGS,
                     datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION)

    client.create_collection(
        collection_name=USER_COLLECTION,
        schema=schema,
        consistency_level="Strong"
    )

    def generate_random_string(length):
        return ''.join(random.choice(string.ascii_letters + string.digits)
                       for _ in range(length))

    def generate_random_entities(num_entities, dim) -> List[Dict[str, Any]]:
        entities_ = []
        for _ in range(num_entities):
            pk = generate_random_string(10)
            random_value = random.random()
            embeddings = np.random.rand(dim).tolist()
            entities_.append({FieldName.PK: pk,
                              FieldName.RANDOM: random_value,
                              FieldName.EMBEDDINGS: embeddings})
        return entities_

    entities = generate_random_entities(NUM_ENTITIES, EMBEDDING_DIMENSION)

    client.insert(collection_name=USER_COLLECTION, data=entities)

    index_params = client.prepare_index_params()

    index_params.add_index(field_name=FieldName.PK)

    index_params.add_index(
        field_name=FieldName.EMBEDDINGS,
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 128}
    )

    client.create_index(
        collection_name=USER_COLLECTION,
        index_params=index_params
    )


def main():
    root_client = MilvusClient(URI, user=ROOT_USER, password=ROOT_PASS)
    remove_user(root_client)
    user_client = add_user(root_client)
    create_collection(user_client)

    print(f'root client list collections: {root_client.list_collections()}')
    print(f'{USERNAME} client list collections: {user_client.list_collections()}')


if __name__ == '__main__':
    main()
