import pandas as pd

# CSV 파일 불러오기
file_path = '/home/songmu/Multipath/python/logs/packets_log.csv'
df = pd.read_csv(file_path)

# Interface IP와 Interface ID 별로 데이터 분할
df_1 = df[df['Interface ID'] == 1]
df_2 = df[df['Interface ID'] == 2]

# Sequence Number 기준으로 병합
merged_df = pd.merge(df_1, df_2, on='Sequence Number', suffixes=('_1', '_2'), how='outer')

# 각 상황에 대한 분석
missing_1 = merged_df[merged_df['Interface IP_1'].isna()]
missing_2 = merged_df[merged_df['Interface IP_2'].isna()]

latency_difference = merged_df[~merged_df['Latency_1'].isna() & ~merged_df['Latency_2'].isna()]
latency_difference['Latency Difference'] = abs(latency_difference['Latency_1'] - latency_difference['Latency_2'])

# 차이가 큰 패킷만 필터링 (예: Latency 차이가 0.1초 (100ms) 이상인 경우)
significant_latency_diff = latency_difference[latency_difference['Latency Difference'] > 0.02]

# 누락된 Sequence Number 분석
all_sequence_numbers = set(range(int(df['Sequence Number'].min()), int(df['Sequence Number'].max()) + 1))

received_sequence_numbers_1 = set(df_1['Sequence Number'])
received_sequence_numbers_2 = set(df_2['Sequence Number'])

missing_sequence_numbers_1 = sorted(all_sequence_numbers - received_sequence_numbers_1)
missing_sequence_numbers_2 = sorted(all_sequence_numbers - received_sequence_numbers_2)

# Latency가 50ms를 초과하는 패킷 분석
latency_exceeded_1 = df_1[df_1['Latency'] > 0.05]
latency_exceeded_2 = df_2[df_2['Latency'] > 0.05]

# Loss 패킷 총 개수
total_loss_1 = len(missing_sequence_numbers_1) + len(latency_exceeded_1)
total_loss_2 = len(missing_sequence_numbers_2) + len(latency_exceeded_2)

# 다른 인터페이스에서 커버할 수 있는지 분석
covered_by_other_interface = pd.merge(latency_exceeded_1, df_2, on='Sequence Number', suffixes=('_1', '_2'))
covered_by_other_interface = covered_by_other_interface[covered_by_other_interface['Latency_2'] <= 0.05]

covered_by_other_interface_2 = pd.merge(latency_exceeded_2, df_1, on='Sequence Number', suffixes=('_2', '_1'))
covered_by_other_interface_2 = covered_by_other_interface_2[covered_by_other_interface_2['Latency_1'] <= 0.05]

# 결과 저장
missing_1.to_csv('/home/songmu/Multipath/python/logs/missing_1.csv', index=False)
missing_2.to_csv('/home/songmu/Multipath/python/logs/missing_2.csv', index=False)
significant_latency_diff.to_csv('/home/songmu/Multipath/python/logs/significant_latency_diff.csv', index=False)

latency_exceeded_1.to_csv('/home/songmu/Multipath/python/logs/latency_exceeded_1.csv', index=False)
latency_exceeded_2.to_csv('/home/songmu/Multipath/python/logs/latency_exceeded_2.csv', index=False)

# 누락된 Sequence Number를 CSV로 저장
missing_seq_1_df = pd.DataFrame(missing_sequence_numbers_1, columns=['Missing Sequence Number'])
missing_seq_1_df.to_csv('/home/songmu/Multipath/python/logs/missing_sequence_numbers_1.csv', index=False)

missing_seq_2_df = pd.DataFrame(missing_sequence_numbers_2, columns=['Missing Sequence Number'])
missing_seq_2_df.to_csv('/home/songmu/Multipath/python/logs/missing_sequence_numbers_2.csv', index=False)

# 총 Loss 패킷 개수 저장
with open('/home/songmu/Multipath/python/logs/total_loss.txt', 'w') as f:
    f.write(f'Total Loss in Interface 1: {total_loss_1}\n')
    f.write(f'Total Loss in Interface 2: {total_loss_2}\n')

# 다른 인터페이스에서 커버 가능한 패킷 저장
covered_by_other_interface.to_csv('/home/songmu/Multipath/python/logs/covered_by_other_interface.csv', index=False)
covered_by_other_interface_2.to_csv('/home/songmu/Multipath/python/logs/covered_by_other_interface_2.csv', index=False)
