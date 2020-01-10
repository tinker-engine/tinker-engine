import json
import requests
from ..problem import *


problem_json = json.dumps(
    {'task_metadata': {
        'adaptation_can_use_pretrained_model': False,
        'adaptation_dataset': 'mnist',
        'adaptation_evaluation_metrics': ['accuracy'],
        'adaptation_label_budget': [5, 2000, 3000],
        'base_can_use_pretrained_model': True,
        'base_dataset': 'mnist',
        'base_evaluation_metrics': ['accuracy'],
        'base_label_budget': [3000, 6000, 8000],
        'proble_id': 'problem_test'}
    }
)

token_json = json.dumps({'session_token': 'BZgFbWCDcAsOoXGjcNCX'})

status_json = json.dumps(
     {'Session_Status': {'active': True,
                         'budget_left_until_checkpoint': 3000,
                         'current_dataset': {'data_url': '/datasets/lwll_datasets/mnist/mnist_sample/train',
                                             'dataset_type': 'image_classification',
                                             'name': 'mnist',
                                             'number_of_classes': 10,
                                             'number_of_samples_test': 1000,
                                             'number_of_samples_train': 5000,
                                             'uid': 'mnist'},
                         'current_label_budget_stages': [3000, 6000, 8000],
                         'date_created': 1574396697000,
                         'date_last_interacted': 1574396697000,
                         'pair_stage': 'base',
                         'task_id': 'problem_test',
                         'uid': 'BZgFbWCDcAsOoXGjcNCX',
                         'user_name': 'DEMO_TEAM',
                         'using_sample_datasets': True}}
)

seed_labels_json = json.dumps(
    {'Labels': [{'id': '56847.png', 'label': '2'},
                {'id': '45781.png', 'label': '3'},
                {'id': '40214.png', 'label': '7'},
                {'id': '49851.png', 'label': '8'},
                {'id': '46024.png', 'label': '6'},
                {'id': '13748.png', 'label': '1'},
                {'id': '13247.png', 'label': '9'},
                {'id': '39791.png', 'label': '4'},
                {'id': '37059.png', 'label': '0'},
                {'id': '46244.png', 'label': '5'}]}
)


def get_problem(requests_mock):
    mocked_url = 'http://test.com'
    requests_mock.get(
        '{}/task_metadata/problem_id'.format(mocked_url),
        text=problem_json)

    requests_mock.get(
        '{}/auth/get_session_token/full/problem_id'.format(mocked_url),
        text=token_json)

    requests_mock.get(
        '{}/session_status'.format(mocked_url),
        text=status_json
    )
    requests_mock.get(
        '{}/seed_labels'.format(mocked_url),
        text=seed_labels_json
    )
    return LwLL('secret', mocked_url, 'problem_id')


def test_lwll_initialization(requests_mock):
    get_problem(requests_mock)
