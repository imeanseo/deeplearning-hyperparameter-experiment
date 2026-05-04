import pandas as pd
import re
from collections import Counter
import numpy as np

train = pd.read_csv('data/train_df.csv')

print("=" * 80)
print("🔍 오분류 원인 심층 분석")
print("=" * 80)

# 1. 혼동 가능한 패턴들
print("\n1️⃣ Negative인데 긍정 키워드 포함된 사례 (false positive 원인)")
neg = train[train['label'] == 0]
pos_keywords = ['love', 'good', 'great', 'happy', 'like', 'amazing', 'nice', 'yay', 'lol']
ambiguous_neg = neg[neg['text'].str.lower().str.contains('|'.join(pos_keywords), na=False)]
print(f"   {len(ambiguous_neg)}개 ({len(ambiguous_neg)/len(neg)*100:.1f}%)")
for t in ambiguous_neg['text'].head(5).tolist():
    print(f"   → {t[:100]}")

print("\n2️⃣ Positive인데 부정 키워드 포함된 사례 (false negative 원인)")
pos = train[train['label'] == 2]
neg_keywords = ['not', 'no', 'hate', 'bad', 'sucks', "don't", "can't", 'never']
ambiguous_pos = pos[pos['text'].str.lower().str.contains('|'.join(neg_keywords), na=False)]
print(f"   {len(ambiguous_pos)}개 ({len(ambiguous_pos)/len(pos)*100:.1f}%)")
for t in ambiguous_pos['text'].head(5).tolist():
    print(f"   → {t[:100]}")

print("\n3️⃣ Neutral인데 감성 키워드 없는 사례 (neutral 분류 패턴)")
neu = train[train['label'] == 1]
all_sentiment_kw = pos_keywords + ['hate', 'bad', 'sucks', 'awful', 'terrible', 'sad', 'angry']
pure_neutral = neu[~neu['text'].str.lower().str.contains('|'.join(all_sentiment_kw), na=False)]
print(f"   순수 중립 (감성 키워드 없음): {len(pure_neutral)}개 ({len(pure_neutral)/len(neu)*100:.1f}%)")

print("\n" + "=" * 80)
print("🔤 형태소/토큰 분석")
print("=" * 80)

print("\n4️⃣ 대문자 반복 패턴 (강조 표현)")
patterns = {
    'HAHA/HEHE 류': r'\b(haha+|hehe+|lmao|rofl)\b',
    '늘임표 (!!!)': r'!{2,}',
    '... 줄임표': r'\.{3,}',
    '반복 문자 (eeeee)': r'([a-z])\1{3,}',
    'URL 포함': r'http\S+',
    '숫자만': r'^\d+$',
    '이모티콘 :) :(': r'[:;]-?[)(D]|[)(D]-?[:;]',
}
for name, pattern in patterns.items():
    count = sum(1 for t in train['text'] if re.search(pattern, str(t).lower()))
    pct = count/len(train)*100
    print(f"   '{name}': {count}개 ({pct:.1f}%)")

print("\n5️⃣ 레이블별 특수 패턴 비율")
for label, sentiment in [(0,'negative'), (1,'neutral'), (2,'positive')]:
    subset = train[train['label'] == label]
    excl = subset['text'].str.count('!').mean()
    ques = subset['text'].str.count('\?').mean()
    caps = subset['text'].apply(lambda t: len([w for w in str(t).split() if w.isupper() and len(w)>1])).mean()
    url  = subset['text'].str.contains('http', na=False).mean()
    print(f"   {sentiment}: 느낌표={excl:.3f} | 물음표={ques:.3f} | CAPS={caps:.3f} | URL={url:.3f}")

print("\n" + "=" * 80)
print("💡 미처리된 패턴 탐색")
print("=" * 80)

print("\n6️⃣ 여전히 백틱 포함된 고빈도 패턴")
backtick_texts = train[train['text'].str.contains('`', na=False)]['text']
patterns_bt = Counter()
for t in backtick_texts:
    matches = re.findall(r"\w+`\w+", str(t))
    for m in matches:
        patterns_bt[m] += 1
print("   TOP 20 백틱 패턴:")
for pat, cnt in patterns_bt.most_common(20):
    print(f"   '{pat}': {cnt}회")

print("\n7️⃣ 앱 리뷰 vs SNS 트윗 분류 (데이터 출처 혼합)")
app_review_kw = ['app', 'feature', 'update', 'version', 'download', 'install', 'review', 'star', 'rating']
app_review_count = sum(1 for t in train['text'] if any(k in str(t).lower() for k in app_review_kw))
print(f"   앱 리뷰 스타일: {app_review_count}개 ({app_review_count/len(train)*100:.1f}%)")
print(f"   SNS 트윗 스타일: {len(train)-app_review_count}개 ({(1-app_review_count/len(train))*100:.1f}%)")

# 앱 리뷰 vs 트윗 레이블 분포 비교
app_df = train[train['text'].str.lower().str.contains('|'.join(app_review_kw), na=False)]
tweet_df = train[~train['text'].str.lower().str.contains('|'.join(app_review_kw), na=False)]
print(f"\n   앱 리뷰 레이블 분포: neg={app_df['label'].eq(0).mean():.2f} | neu={app_df['label'].eq(1).mean():.2f} | pos={app_df['label'].eq(2).mean():.2f}")
print(f"   트윗 레이블 분포:     neg={tweet_df['label'].eq(0).mean():.2f} | neu={tweet_df['label'].eq(1).mean():.2f} | pos={tweet_df['label'].eq(2).mean():.2f}")

print("\n8️⃣ 추가 슬랭 탐색")
extra_slangs = ['ugh', 'eww', 'ugh', 'meh', 'smh', 'tbh', 'imo', 'fyi', 'irl', 'rn', 'ngl', 'hmu', 'ily', 'omfg', 'af', 'brb', 'thx', 'tho', 'tbt', 'bc', 'cuz', 'coz', 'kinda', 'sorta', 'yep', 'nope', 'yup', 'yeh', 'nah', 'soo', 'sooo']
print("   발견된 추가 슬랭:")
found = []
for slang in extra_slangs:
    count = sum(1 for t in train['text'] if re.search(r'\b' + slang + r'\b', str(t).lower()))
    if count >= 30:
        found.append((slang, count))
for s, c in sorted(found, key=lambda x: -x[1]):
    print(f"   '{s}': {c}회")

print("\n9️⃣ 길이별 레이블 분포 (짧은 텍스트 패턴)")
train['word_count'] = train['text'].str.split().str.len()
for bucket, (lo, hi) in [('초단문 (1~5단어)', (1,5)), ('단문 (6~15단어)', (6,15)), ('중문 (16~30단어)', (16,30)), ('장문 (31+단어)', (31,9999))]:
    subset = train[(train['word_count'] >= lo) & (train['word_count'] <= hi)]
    neg_r = subset['label'].eq(0).mean()
    neu_r = subset['label'].eq(1).mean()
    pos_r = subset['label'].eq(2).mean()
    print(f"   {bucket}: {len(subset)}개 | neg={neg_r:.2f} | neu={neu_r:.2f} | pos={pos_r:.2f}")
