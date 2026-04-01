import yaml
import numpy as np
import pandas as pd
from datetime import timedelta

# 1. Config 로드 및 시드 고정
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

np.random.seed(config['simulation']['random_seed'])

def generate_products(cfg):
    """5만개 상품 생성: 가격대(price_tier) 속성 추가"""
    products = []
    cat_tree = cfg['categories']
    flat_cats = [(d1, d2, d3) for d1, d2_dict in cat_tree.items() 
                              for d2, d3_list in d2_dict.items() 
                              for d3 in d3_list]
    
    for i in range(cfg['simulation']['num_products']):
        d1, d2, d3 = flat_cats[np.random.choice(len(flat_cats))]
        price = int(np.random.exponential(50000) // 1000 * 1000) + 5000
        # 가격 민감도 맵핑을 위한 Tier 분류 (임의 기준)
        if price < 30000: price_tier = 'low'
        elif price < 80000: price_tier = 'medium'
        else: price_tier = 'high'
            
        products.append({
            'product_id': f"P{str(i).zfill(6)}",
            'category_L1': d1,
            'category_L2': d2,
            'category_L3': d3,
            'price': price,
            'price_tier': price_tier
        })
    return pd.DataFrame(products)

def generate_users(cfg):
    """1만명 고객 생성 및 페르소나 맵핑"""
    personas = list(cfg['personas'].keys())
    weights = [v['weight'] for v in cfg['personas'].values()]
    
    users = []
    for i in range(cfg['simulation']['num_users']):
        persona = np.random.choice(personas, p=weights)
        users.append({'user_id': f"U{str(i).zfill(5)}", 'persona': persona})
    return pd.DataFrame(users)

def generate_logs_markov(cfg, users_df, products_df):
    """Markov Chain 및 페르소나 설정이 완벽히 반영된 100만건 행동 로그 생성"""
    num_logs = cfg['simulation']['num_logs']
    
    # 빠른 맵핑을 위한 딕셔너리 생성
    user_persona_map = dict(zip(users_df['user_id'], users_df['persona']))
    persona_configs = cfg['personas']
    transition_probs = cfg['transition_probs']
    
    # 상품을 카테고리별로 미리 인덱싱 (빠른 샘플링 용도)
    prod_by_cat = products_df.groupby('category_L1')['product_id'].apply(list).to_dict()
    all_products = products_df['product_id'].tolist()
    
    logs = []
    current_time = pd.to_datetime(cfg['simulation']['start_date'])
    user_ids = users_df['user_id'].values
    
    print("정교한 세션 기반 행동 로그 생성 중... (Markov Chain)")
    
    while len(logs) < num_logs:
        # 1. 세션 시작: 임의의 유저 선택
        user_id = np.random.choice(user_ids)
        persona_name = user_persona_map[user_id]
        p_config = persona_configs[persona_name]
        
        # 2. 페르소나에 따른 타겟 상품군 결정
        # 80% 확률로 선호 카테고리에서 탐색, 20%는 다른 카테고리 탐색
        if np.random.rand() < 0.8 and p_config['preferred_categories']:
            pref_cat = np.random.choice(p_config['preferred_categories'])
            target_pool = prod_by_cat.get(pref_cat, all_products)
        else:
            target_pool = all_products
            
        # 세션에서 기준이 될 상품 1개 선택
        current_product = np.random.choice(target_pool)
        
        # 3. Markov Chain 상태 전이 시작 (초기 상태는 무조건 search 또는 view)
        state = np.random.choice(['search', 'view'], p=[0.7, 0.3])
        
        while state != 'exit' and len(logs) < num_logs:
            logs.append({
                'user_id': user_id,
                'product_id': current_product,
                'event_type': state,
                'timestamp': current_time
            })
            
            # 시간 증가 (이벤트 간 10초 ~ 3분 간격)
            current_time += timedelta(seconds=np.random.randint(10, 180))
            
            # 다음 상태 결정
            trans_map = transition_probs[state]
            next_states = list(trans_map.keys())
            probs = list(trans_map.values())
            
            # [핵심] 전환율(CVR) 차별화 로직 적용
            # cart에서 purchase로 갈 때 페르소나의 conversion_rate_multiplier 적용
            if state == 'cart' and 'purchase' in next_states:
                idx = next_states.index('purchase')
                base_prob = probs[idx]
                adjusted_prob = base_prob * p_config['conversion_rate_multiplier']
                
                # 확률 재정규화 (합이 1이 되도록)
                probs[idx] = min(0.99, adjusted_prob) 
                diff = probs[idx] - base_prob
                # 남은 확률을 exit에서 빼거나 더함
                exit_idx = next_states.index('exit')
                probs[exit_idx] -= diff
                
                # 오류 방지 정규화
                probs = np.array(probs)
                probs = np.clip(probs, 0, 1)
                probs = probs / probs.sum()

            state = np.random.choice(next_states, p=probs)
            
            # view나 search 상태를 반복할 때 간혹 상품을 변경함
            if state in ['search', 'view'] and np.random.rand() < 0.4:
                 current_product = np.random.choice(target_pool)
                 
    # 로그 생성 완료 후 시간순 정렬 (전체 시스템 로딩 순서 시뮬레이션)
    df_logs = pd.DataFrame(logs)
    df_logs = df_logs.sort_values('timestamp').reset_index(drop=True)
    return df_logs

def split_and_save(products, users, logs):
    """시간 기반 Train/Valid/Test (8:1:1) 분할 및 저장"""
    n_total = len(logs)
    train_idx = int(n_total * 0.8)
    valid_idx = int(n_total * 0.9)
    
    train_logs = logs.iloc[:train_idx]
    valid_logs = logs.iloc[train_idx:valid_idx]
    test_logs = logs.iloc[valid_idx:]
    
    import os
    os.makedirs('data', exist_ok=True)
    products.to_csv('data/products.csv', index=False)
    users.to_csv('data/users.csv', index=False)
    train_logs.to_csv('data/train_logs.csv', index=False)
    valid_logs.to_csv('data/valid_logs.csv', index=False)
    test_logs.to_csv('data/test_logs.csv', index=False)

if __name__ == "__main__":
    df_prod = generate_products(config)
    df_user = generate_users(config)
    df_log = generate_logs_markov(config, df_user, df_prod)
    
    split_and_save(df_prod, df_user, df_log)