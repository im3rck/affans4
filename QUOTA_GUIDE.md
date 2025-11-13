# API Quota Management Guide

## 🎯 Understanding API Usage

### API Calls Per Query

| Configuration | API Calls | Free Tier Safe? |
|--------------|-----------|-----------------|
| **Minimal** (No query rewriting) | 1 | ✅ Yes (~1500 queries/day) |
| **Balanced** (Query rewriting ON) | 2-3 | ✅ Yes (~500 queries/day) |
| **Full Featured** (All features) | 3-4 | ⚠️ Moderate (~375 queries/day) |
| **Clustering** (Per cluster) | 1 each | ⚠️ Use sparingly |

### Google Gemini Free Tier Limits (2024)

**Gemini 1.5 Pro:**
- 15 requests per minute (RPM)
- ~1,500 requests per day
- Free tier

**Gemini 1.5 Flash:**
- 15 requests per minute (RPM)
- Higher throughput
- Free tier

**Gemini Pro:**
- 60 requests per minute (RPM)
- Lower daily limit
- Free tier

## 🚀 Quick Start: Choose Your Profile

### Option 1: FREE_TIER Profile (Recommended for Testing)

**Best for:** Testing, learning, low-volume usage

```python
# In config.py, set:
ACTIVE_PROFILE = "FREE_TIER"
```

**Features:**
- ✅ Reranking (doesn't use Gemini API)
- ❌ Query rewriting (disabled)
- ❌ Clustering (disabled)
- ❌ HyDE (disabled)

**API Usage:** **1 call per query**
**Daily capacity:** ~1,500 queries

**Run with:**
```bash
python config.py  # View settings
streamlit run app_optimized.py  # Optimized app with tracking
```

### Option 2: BALANCED Profile (Recommended for Most Users)

**Best for:** Good performance with moderate API usage

```python
# In config.py, set:
ACTIVE_PROFILE = "BALANCED"
```

**Features:**
- ✅ Reranking
- ✅ Query rewriting (limited to 2 variations)
- ✅ Clustering (max 3 clusters)
- ❌ HyDE (disabled)

**API Usage:** **2-3 calls per query**
**Daily capacity:** ~500-750 queries

### Option 3: FULL_FEATURED Profile (For Paid API Keys)

**Best for:** Production use with paid API key

```python
ACTIVE_PROFILE = "FULL_FEATURED"
```

**Features:**
- ✅ All features enabled
- ✅ Best quality answers

**API Usage:** **3-4 calls per query**
**Daily capacity:** ~375-500 queries (free tier)

## 💡 Tips to Avoid Quota Issues

### 1. Use the Optimized App

```bash
# Use this instead of app.py
streamlit run app_optimized.py
```

This app:
- ✅ Tracks API usage in real-time
- ✅ Shows remaining quota
- ✅ Prevents rate limit errors
- ✅ Displays API calls per query

### 2. Disable Expensive Features

**Query Rewriting** uses **+1-2 API calls**:
```python
chatbot = RAGChatbot(
    enable_query_rewriting=False  # Saves 1-2 calls
)
```

**HyDE** uses **+1 API call**:
```python
response = chatbot.answer_query(
    query="...",
    use_hyde=False  # Saves 1 call
)
```

**Clustering** uses **1 call per cluster**:
```python
chatbot = RAGChatbot(
    enable_clustering=False  # Avoid if not needed
)
```

### 3. Use Faster/Cheaper Models

```python
# Gemini 1.5 Flash - fastest, good for high volume
chatbot = RAGChatbot(gemini_model="gemini-1.5-flash")

# Gemini 1.5 Pro - best quality
chatbot = RAGChatbot(gemini_model="gemini-1.5-pro")
```

### 4. Implement Caching (Coming Soon)

Cache responses to avoid re-processing same queries:
```python
# Future feature
chatbot.enable_cache(ttl=3600)  # 1 hour cache
```

### 5. Batch Processing

For large datasets, process in batches:
```python
# Instead of indexing all at once
for batch in document_batches:
    chatbot.index_documents(batch)
    time.sleep(5)  # Avoid rate limits
```

## 📊 Monitor Your Usage

### Check Estimate Before Running

```bash
python config.py
```

Output:
```
📊 Usage Estimation:
============================================================

10 queries/day:
  Total API calls: 10
  % of free tier: 0.7%
  Will exceed: ✅ NO

100 queries/day:
  Total API calls: 100
  % of free tier: 6.7%
  Will exceed: ✅ NO

500 queries/day:
  Total API calls: 500
  % of free tier: 33.3%
  Will exceed: ✅ NO
```

### Real-time Tracking

The optimized app shows:
- API calls made (session)
- Calls in last minute
- Remaining quota
- Estimated calls per query

## 🚨 What to Do If You Hit the Limit

### Error: "Quota exceeded"

**Short-term solution:**
1. Wait 1 minute (for RPM limit)
2. Wait until next day (for daily limit)
3. Switch to FREE_TIER profile

**Long-term solution:**
1. Upgrade to paid tier
2. Implement caching
3. Reduce features
4. Optimize queries

### Error: "Rate limit exceeded"

```python
# Add delay between requests
import time

for query in queries:
    response = chatbot.answer_query(query)
    time.sleep(5)  # Wait 5 seconds between queries
```

## 💰 Upgrading to Paid Tier

### Google AI Studio (Pay-as-you-go)

**Pricing (approx):**
- Gemini 1.5 Pro: $0.00025 per 1K characters input
- Very affordable for most use cases

**Benefits:**
- Higher rate limits (1000+ RPM)
- No daily caps
- Production-ready

**How to upgrade:**
1. Go to [Google AI Studio](https://ai.google.dev/)
2. Set up billing
3. Generate new API key
4. Update your `.env` file

### Cost Estimation

```python
# Assuming average query + context = 2K chars
# Average answer = 500 chars

queries_per_month = 10000
cost_per_query = 0.00025 * 2.5  # Input + output
monthly_cost = queries_per_month * cost_per_query

# ~$6.25/month for 10,000 queries
```

## 🎓 Recommended Configurations

### For Learning/Testing (Free Tier)
```python
ACTIVE_PROFILE = "FREE_TIER"
# Use: streamlit run app_optimized.py
# Capacity: 1,500 queries/day
```

### For Demo/Prototype (Free Tier)
```python
ACTIVE_PROFILE = "BALANCED"
# Moderate features, good performance
# Capacity: 500-750 queries/day
```

### For Production (Paid)
```python
ACTIVE_PROFILE = "PRODUCTION"
# All features enabled
# Implement caching and monitoring
```

## 📋 Quick Reference

| Scenario | Profile | App | Expected Usage |
|----------|---------|-----|----------------|
| Just testing | FREE_TIER | app_optimized.py | <100 queries/day |
| Building demo | BALANCED | app_optimized.py | <500 queries/day |
| Production free | FREE_TIER | app_optimized.py | Monitor closely |
| Production paid | PRODUCTION | app.py or custom | Unlimited |

## ✅ Best Practices Checklist

- [ ] Start with FREE_TIER profile
- [ ] Use `app_optimized.py` for quota tracking
- [ ] Monitor API usage in sidebar
- [ ] Disable query rewriting if not needed
- [ ] Don't use clustering on every query
- [ ] Test with small document sets first
- [ ] Check `python config.py` estimates
- [ ] Upgrade to paid tier for production
- [ ] Implement caching (future)
- [ ] Use faster models when possible

## 🔗 Resources

- [Google AI Studio](https://ai.google.dev/)
- [Gemini API Docs](https://ai.google.dev/docs)
- [Pricing Information](https://ai.google.dev/pricing)
- [Rate Limits Guide](https://ai.google.dev/docs/rate_limits)

---

**Summary:** For free tier usage, use `FREE_TIER` or `BALANCED` profile with `app_optimized.py`. You'll be safe with hundreds of queries per day. For production, upgrade to paid tier (~$6-10/month for typical usage).
