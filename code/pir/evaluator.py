
import os
import json
import re
import time

import drivers
import ttp_utils
import etbc

def evaluate_aql(aql):
    start = time.time()
    res = drivers.fetch(aql, reduce='row')
    exec_end = time.time()

    res = sorted(res)

    return {
        'result': res,
        'result_count': len(res),
        'execution_time': exec_end - start
    }

def evaluate_ecql(ecql):
    start = time.time()
    raw = ecql
    sub_queries = []
    expr = None
    while True:
        m = re.search(r'= \{([^\{\}]+)\}', ecql)
        if not m:
            sub_queries.append(ecql.strip())
            expr = ecql
            break
        ecql = ecql[:m.span()[0]] + 'IN #' + str(len(sub_queries)) + ecql[m.span()[1]:]
        sub_queries.append(m.group(1).strip())

    n = len(sub_queries)

    parse_end = time.time()

    for i in range(n - 1):
        r = drivers.fetch(sub_queries[i], reduce='str')
        for j in range(i + 1, n):
            scope = 'R' if sub_queries[j].startswith('SELECT') else 'G'
            if scope == 'R':
                result = '(' + r + ')'
            elif scope == 'G':
                result = '[' + r + ']'
            sub_queries[j] = sub_queries[j].replace(f'#{i}', result)

    res = drivers.fetch(sub_queries[n - 1], reduce='row')
    res = sorted(res)

    exec_end = time.time()

    return {
        'result': res,
        'result_count': len(res),
        'parse_time': parse_end - start,
        'execution_time': exec_end - parse_end
    }


def test_all_queries(ql, temp_fp):
    with open(temp_fp, 'r') as f:
        data = json.load(f)

    def save():
        with open(temp_fp, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    qlkey = ttp_utils.QL_TYPENAMES[ql]
    evaluate_func = eval(f'evaluate_{qlkey}')
    for i, q in enumerate(data):
        if f'result_{qlkey}' in q:
            continue
        print(f'Evaluating {i} / {len(data)}, {q["question"]}', flush=True)
        try:
            res = evaluate_func(q[qlkey])
            if res[f'result_count'] <= ttp_utils.MAX_CMP_RESULT_COUNT:
                q[f'result_{qlkey}'] = res['result']
            else:
                q[f'result_{qlkey}'] = []
            q[f'result_count_{qlkey}'] = res['result_count']
            q[f'execution_time_{qlkey}'] = res['execution_time']
        except Exception as e:
            save()
            print(f'Error in {i}, {str(e)}')
    save()

    for sample in data:
        if sample['result_count_ecql'] != sample['result_count_aql'] or (sample['result_count_ecql'] == 1 and sample['result_ecql'][0] != sample['result_aql'][0]):
            print(sample['question'], '|', sample['result_count_ecql'], sample['result_count_aql'], sample['result_ecql'][0] if sample['result_ecql'] else '-', sample['result_aql'][0] if sample['result_aql'] else '-')


def question_match(q1, q2):
    return q1.strip().replace(' ', '') == q2.strip().replace(' ', '')


def logical_form_match(gt, q):
    gt = gt.split('|')[-1].strip()
    q = q.split('|')[-1].strip()
    return gt.replace(' ', '') == q.replace(' ', '')


def exec_match(gt, q, ql):
    qlkey = ttp_utils.QL_TYPENAMES[ql]
    evaluate_func = eval(f'evaluate_{qlkey}')
    try:
        res = evaluate_func(q)
        if gt['result_count_' + qlkey] > ttp_utils.MAX_CMP_RESULT_COUNT:
            return True, gt['result_count_' + qlkey] == res['result_count']
        else:
            return True, gt['result_count_' + qlkey] == res['result_count'] and set(gt['result_' + qlkey]) == set(res['result'])
    except Exception as e:
        return False, False


def evaluate_prediction(questions, outputs, ql, tokenizer, pir=False):
    batch_size = outputs.shape[0]
    num_return_sequences = outputs.shape[1]

    predictions = []

    if pir:
        key = ttp_utils.QL_TYPENAMES[ql] + '_pir'
    else:
        key = ttp_utils.QL_TYPENAMES[ql]

    for i in range(batch_size):
        gt = ttp_utils.gt_of(questions[i])
        sample = ttp_utils.sample_of(questions[i])
        candidates = []
        matched = False
        first_matched = False
        for seq_id in range(num_return_sequences):
            pred_q = tokenizer.decode(outputs[i, seq_id, :], skip_special_tokens = True)
            if ql == 'AQL':
                pred_q = pred_q.replace('IN BOUND', 'INBOUND').replace('OUT BOUND', 'OUTBOUND')
                pred_q = pred_q.replace('COLLECT ION', 'COLLECTION')
            candidates.append(pred_q)

            gt_q = gt[key]
            if logical_form_match(gt_q, pred_q):
                # first match
                if seq_id == 0:
                    first_matched = True

                # general match
                matched = True

        predictions.append({
            'id': sample['id'],
            'nlq': questions[i],
            'candidates': candidates,
            'gt': gt_q,
            'matched': matched,
            'first_matched': first_matched
        })

    return predictions


def test_one(gt, pred, ql):
    first_lf, first_ex, guide_lf, guide_ex = False, False, False, False
    qlkey = ttp_utils.QL_TYPENAMES[ql]
    gt_q = gt[qlkey].split('|')[-1].strip()
    
    lf, ex = -1, -1
    for i, cand in enumerate(pred['candidates']):
        q = cand.split('|')[-1].strip()
        print(f'> CANDIDATE {i}: {q}')
        lf_q = logical_form_match(gt_q, q)
        if lf_q:
            lf = i
            break
        else:
            ex_complete, m = exec_match(gt, q, ql)
            if m:
                ex = i
                break
            elif ex_complete:
                break
    
    first_lf = lf == 0
    first_ex = ex == 0
    guide_lf = lf > 0
    guide_ex = ex > 0

    return first_lf, first_ex, guide_lf, guide_ex


def print_pir(q):
    def recursive_print_ecql(q, indent):
        q = q.strip()
        m = re.search(r'\{(.*)\}', q)
        if m:
            print(' ' * indent + q[:m.span()[0] + 1])
            recursive_print_ecql(m.group(1), indent + 2)
            print(' ' * indent + q[m.span()[1] - 1:])
        else:
            print(' ' * indent + q)

    q = q.split('|')[-1].strip()
    # AQL
    if q.startswith('FOR'):
        print(q)
        print()
    # PIR
    elif q.startswith('SCAN'):
        for clause in q.split(';'):
            clause = clause.strip()
            if clause:
                print(clause)
        print()
    # ECQL
    elif q.startswith('SELECT') or q.startswith('MATCH'):
        recursive_print_ecql(q, 0)
    else:
        for clause in q.split(';'):
            clause = clause.strip()
            if clause:
                print(clause)
        print()


def read_train_log(summary_dir, pir=False, eval_folds=[1,2,3,4]):
    exp_name = summary_dir.split("/")[-1]
    print(f'Result of [{exp_name}]')
    for ql, qlkey in ttp_utils.QL_TYPENAMES.items():
        if ql == 'PIR':
            continue
        if ql.lower() in exp_name.lower() or qlkey.lower() in exp_name.lower():
            print(f'QL: {ql} {qlkey}')
            break
    else:
        print('Warning: no matched QL Type.')

    total_summary = {
        'folds': [],
        'train_accuracy': 0,
        'valid_accuracy': 0,
        'train_first_accuracy': 0,
        'valid_first_accuracy': 0
    }

    nfolds = len(os.listdir(summary_dir))
    
    for fold_name in os.listdir(summary_dir):
        fold = int(fold_name[-1])
        if fold not in eval_folds:
            continue
        fold_dir = f'{summary_dir}/{fold_name}'
        fold_summary = {period: {
            'fold': fold,
            'epoch_summary': [],
            'n_queries': 0,
            'best_n_matched': 0,
            'best_n_first_matched': 0,
            'best_epoch': 0,
            'accuracy': 0,
            'first_accuracy': 0
        } for period in ['train', 'valid']}

        for period in ['train', 'valid']:
            fsp = fold_summary[period]
            fn_test = os.listdir(fold_dir)[0]
            if 'json' not in fn_test:
                for epoch in os.listdir(fold_dir):
                    epoch = int(epoch)
                    with open(f'{fold_dir}/{epoch}/queries-{period}.aql', 'r') as f:
                        summary = json.load(f)
                        summary['epoch'] = epoch
                        fsp['n_queries'] = summary['n_queries']
                        fsp['epoch_summary'].append(summary)
                        if summary['n_first_matched'] > fsp['best_n_first_matched'] \
                            or summary['n_first_matched'] == fsp['best_n_first_matched'] and summary['n_matched'] > fsp['best_n_matched'] \
                            or summary['n_first_matched'] == fsp['best_n_first_matched'] and summary['n_matched'] == fsp['best_n_matched'] and summary['epoch'] > fsp['best_epoch']:
                            fsp['best_n_first_matched'] = summary['n_first_matched']
                            fsp['best_n_matched'] = summary['n_matched']
                            fsp['best_epoch'] = epoch
            else:
                epochs = sorted(list(set(int(fn.split('-')[0]) for fn in os.listdir(fold_dir))))
                for epoch in epochs:
                    with open(f'{fold_dir}/{epoch}-{period}.json', 'r') as f:
                        summary = json.load(f)
                        summary['epoch'] = epoch
                        fsp['n_queries'] = summary['n_queries']
                        fsp['epoch_summary'].append(summary)
                        if summary['n_first_matched'] > fsp['best_n_first_matched'] \
                            or summary['n_first_matched'] == fsp['best_n_first_matched'] and summary['n_matched'] > fsp['best_n_matched'] \
                            or summary['n_first_matched'] == fsp['best_n_first_matched'] and summary['n_matched'] == fsp['best_n_matched'] and summary['epoch'] > fsp['best_epoch']:
                            fsp['best_n_first_matched'] = summary['n_first_matched']
                            fsp['best_n_matched'] = summary['n_matched']
                            fsp['best_epoch'] = epoch

            fsp['accuracy'] = fsp['best_n_matched'] / fsp['n_queries']
            fsp['first_accuracy'] = fsp['best_n_first_matched'] / fsp['n_queries']
        
        total_summary['folds'].append(fold_summary)
        total_summary['train_accuracy'] += fold_summary['train']['accuracy']
        total_summary['valid_accuracy'] += fold_summary['valid']['accuracy']
        total_summary['train_first_accuracy'] += fold_summary['train']['first_accuracy']
        total_summary['valid_first_accuracy'] += fold_summary['valid']['first_accuracy']

    total_summary['train_accuracy'] /= nfolds
    total_summary['valid_accuracy'] /= nfolds
    total_summary['train_first_accuracy'] /= nfolds
    total_summary['valid_first_accuracy'] /= nfolds

    print(f'Train accuracy: {total_summary["train_accuracy"]:.5f}')
    print(f'Valid accuracy: {total_summary["valid_accuracy"]:.5f}')
    print(f'Train first accuracy: {total_summary["train_first_accuracy"]:.5f}')
    print(f'Valid first accuracy: {total_summary["valid_first_accuracy"]:.5f}')
    
    lfs = []
    exs = []

    lfs_etbc = []
    exs_etbc = []

    for i, fold_summary in enumerate(total_summary['folds']):
        print(f'Fold: {i + 1}')
        # if i != 0:
        #     continue
        for period in ['train', 'valid']:
            fsp = fold_summary[period]
            print(f'{period}: match {fsp["best_n_matched"]} / {fsp["n_queries"]}, first match {fsp["best_n_first_matched"]} / {fsp["n_queries"]} (Epoch {fsp["best_epoch"]})')
        best_epoch = fold_summary['valid']['best_epoch']
        stats_raw = [0, 0, 0, 0]
        stats_etbc = [0, 0, 0, 0]
        n = 0
        for epoch_summary in fold_summary['valid']['epoch_summary']:
            if epoch_summary['epoch'] == best_epoch:
                n = len(epoch_summary['prediction'])
                epoch_data = []
                for pred in epoch_summary['prediction']:
                    gt = ttp_utils.gt_of(pred['nlq'])
                    sample = ttp_utils.sample_of(pred['nlq'])
                    epoch_data.append([pred, gt, sample])
                epoch_data = sorted(epoch_data, key=lambda t: t[2]['id'])

                for pred, gt, sample in epoch_data:
                    # if question_match(pred['nlq'], q):
                    print('-' * 20, sample['id'], sample['template'], '-' * 60)
                    print(pred['nlq'])
                    print()
                    print('[GT] ', end='')
                    print_pir(pred['gt'])

                    if pred['first_matched']:
                        print('[CORRECT]')
                    else:
                        for candidate in pred['candidates']:
                            prefix = '[1] ' if logical_form_match(pred['gt'], candidate) else '[0] '
                            print(prefix, end='')
                            print_pir(candidate)
                    if pir:
                        s_raw = test_one(gt, {'candidates': [etbc.raw_compile(cq) for cq in pred['candidates']]}, ql)
                        s_etbc = test_one(gt, {'candidates': [etbc.backward_compile(cq) for cq in pred['candidates']]}, ql)
                    else:
                        s_raw = test_one(gt, pred, ql)
                        s_etbc = False, False, False, False
                    matched_raw = 'NO'
                    matched_etbc = 'NO'
                    for met, label in enumerate(['FIRST LF', 'FIRST EX', 'GUIDE LF', 'GUIDE EX']):
                        stats_raw[met] += s_raw[met]
                        stats_etbc[met] += s_etbc[met]
                        if s_raw[met]:
                            matched_raw = label
                        if s_etbc[met]:
                            matched_etbc = label
                    print(f'Match Status: [{matched_raw}] [{matched_etbc}]', flush=True)

        
        lf = (stats_raw[0] + stats_raw[2])
        ex = (stats_raw[1] + stats_raw[3])
        lfs.append(lf / n)
        exs.append(ex / n)

        lf_etbc = (stats_etbc[0] + stats_etbc[2])
        ex_etbc = (stats_etbc[1] + stats_etbc[3])
        lfs_etbc.append(lf_etbc / n)
        exs_etbc.append(ex_etbc / n)

        print()
        print(f'LF: {lf} / {n} ({lf / n:.4f}), EX: {lf + ex} / {n} ({(lf + ex) / n:.4f}), LF(ETBC): {lf_etbc} / {n} ({lf_etbc / n:.4f}), EX(ETBC): {lf_etbc + ex_etbc} / {n} ({(lf_etbc + ex_etbc) / n:.4f})')
    
    print(f'Average LF: {sum(lfs) / nfolds:.4f}, EX: {(sum(lfs) + sum(exs)) / nfolds:.4f}')
    print(f'Average LF(ETBC): {sum(lfs_etbc) / nfolds:.4f}, EX(ETBC): {(sum(lfs_etbc) + sum(exs_etbc)) / nfolds:.4f}')

    return total_summary


def test_llm(fp, eval_folds=[1,2,3,4]):
    exp_name = fp.split('/')[-1].split('.')[0]

    for ql, qlkey in ttp_utils.QL_TYPENAMES.items():
        if ql == 'PIR':
            continue
        if ql.lower() in exp_name.lower() or qlkey.lower() in exp_name.lower():
            print(f'QL: {ql} {qlkey}')
            break
    else:
        print('Warning: no matched QL Type.')

    with open(fp, 'r') as f:
        data = json.load(f)

    lfs, exs = [], []
    nfolds = len(eval_folds)

    for i, fold in enumerate(data):
        if (i + 1) not in eval_folds:
            continue
        print(f'Fold: {i + 1}')
        lf, ex = 0, 0
        for sample in fold:
            gt = ttp_utils.gt_of(sample['id'])
            print('-' * 20, sample['id'], gt['template'], '-' * 60)
            print('[GT] ', end='')
            print_pir(gt[qlkey])

            s = test_one(gt, {'candidates': [sample['query']]}, ql)
            if s[0]:
                lf += 1
                label = 'LF'
            elif s[1]:
                ex += 1
                label = 'EX'
            else:
                label = 'NO'
            print(f'Match Status: [{label}]', flush=True)
        n = len(fold)
        lfs.append(lf / n)
        exs.append(ex / n)

        print(f'LF: {lf} / {n} ({lf / n:.4f}), EX: {lf + ex} / {n} ({(lf + ex) / n:.4f})')

    print(f'Average LF: {sum(lfs) / nfolds:.4f}, EX: {(sum(lfs) + sum(exs)) / nfolds:.4f}')
            
