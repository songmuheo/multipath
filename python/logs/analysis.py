
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

# 결과 출력
print("1번 인터페이스에서 수신되지 않은 패킷:")
print(missing_1)

print("\n2번 인터페이스에서 수신되지 않은 패킷:")
print(missing_2)

print("\nLatency 차이가 큰 패킷:")
print(significant_latency_diff)

print("\n1번 인터페이스에서 누락된 Sequence Number:")
print(missing_sequence_numbers_1)

print("\n2번 인터페이스에서 누락된 Sequence Number:")
print(missing_sequence_numbers_2)

# # 결과 저장
# missing_1.to_csv('missing_1.csv', index=False)
# missing_2.to_csv('missing_2.csv', index=False)
# significant_latency_diff.to_csv('significant_latency_diff.csv', index=False)

# # 누락된 Sequence Number를 CSV로 저장
# missing_seq_1_df = pd.DataFrame(missing_sequence_numbers_1, columns=['Missing Sequence Number'])
# missing_seq_1_df.to_csv('missing_sequence_numbers_1.csv', index=False)

# missing_seq_2_df = pd.DataFrame(missing_sequence_numbers_2, columns=['Missing Sequence Number'])
# missing_seq_2_df.to_csv('missing_sequence_numbers_2.csv', index=False)
