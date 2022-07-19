from robustbench.data import load_cifar10
from robustbench.data import load_cifar10c
from robustbench.utils import clean_accuracy
from robustbench.utils import load_model
import foolbox as fb
from autoattack import AutoAttack


def test_model():
    x_test, y_test = load_cifar10(n_examples=50)
    model = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    _, advs, success = fb.attacks.LinfPGD()(fmodel, x_test.to('cuda:0'), y_test.to('cuda:0'), epsilons=[8/255])
    print('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test, y_test)

def corruption():
    corruptions = ['fog']
    x_test, y_test = load_cifar10c(n_examples=1000, corruptions=corruptions, severity=5)
    for model_name in ['Standard', 'Engstrom2019Robustness', 'Rice2020Overfitting',
                    'Carmon2019Unlabeled']:
        model = load_model(model_name, dataset='cifar10', threat_model='Linf')
        acc = clean_accuracy(model, x_test, y_test)
        print(f'Model: {model_name}, CIFAR-10-C accuracy: {acc:.1%}')

if __name__ == '__main__':
    test_model()
    corruption()