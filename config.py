"""
Configuration for RAG Chatbot
Adjust these settings based on your API quota and performance needs
"""

# ============================================================================
# CONFIGURATION PROFILES
# ============================================================================

# Profile 1: FREE TIER OPTIMIZED (Minimal API usage)
FREE_TIER_CONFIG = {
    "gemini_model": "gemini-1.5-flash-latest",  # Fast and efficient
    "enable_query_rewriting": False,  # DISABLED - saves 1-2 API calls per query
    "enable_reranking": True,  # Keep this - doesn't use Gemini API
    "enable_clustering": False,  # DISABLED - can use many API calls
    "use_hyde": False,  # DISABLED - saves 1 API call
    "cache_enabled": True,  # Enable caching to reuse results
}

# Profile 2: BALANCED (Good performance, moderate API usage)
BALANCED_CONFIG = {
    "gemini_model": "gemini-1.5-flash-latest",  # Faster than Pro, good balance
    "enable_query_rewriting": True,  # Enabled but limited
    "max_query_variations": 2,  # Limit variations (default is 3)
    "enable_reranking": True,
    "enable_clustering": True,
    "max_clusters": 3,  # Limit cluster analysis calls
    "use_hyde": False,
    "cache_enabled": True,
}

# Profile 3: FULL FEATURED (Best quality, high API usage)
FULL_FEATURED_CONFIG = {
    "gemini_model": "gemini-1.5-pro-latest",
    "enable_query_rewriting": True,
    "enable_reranking": True,
    "enable_clustering": True,
    "use_hyde": True,  # Best retrieval quality
    "cache_enabled": True,
}

# Profile 4: PRODUCTION (Paid API key)
PRODUCTION_CONFIG = {
    "gemini_model": "gemini-1.5-pro-latest",
    "enable_query_rewriting": True,
    "enable_reranking": True,
    "enable_clustering": True,
    "use_hyde": True,
    "cache_enabled": True,
    "rate_limit_rpm": 1000,  # Set based on your paid tier
    "enable_exponential_backoff": True,
}

# ============================================================================
# SELECT YOUR PROFILE HERE
# ============================================================================

# Change this to switch profiles
ACTIVE_PROFILE = "FREE_TIER"  # Options: FREE_TIER, BALANCED, FULL_FEATURED, PRODUCTION

# Get active config
CONFIG_MAP = {
    "FREE_TIER": FREE_TIER_CONFIG,
    "BALANCED": BALANCED_CONFIG,
    "FULL_FEATURED": FULL_FEATURED_CONFIG,
    "PRODUCTION": PRODUCTION_CONFIG,
}

ACTIVE_CONFIG = CONFIG_MAP.get(ACTIVE_PROFILE, FREE_TIER_CONFIG)

# ============================================================================
# USAGE TRACKING
# ============================================================================

class APIUsageTracker:
    """Track API usage to avoid quota exceeded"""

    def __init__(self, max_rpm=15):
        self.max_rpm = max_rpm
        self.calls = []

    def can_make_call(self):
        """Check if we can make an API call without exceeding quota"""
        import time
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]
        return len(self.calls) < self.max_rpm

    def record_call(self):
        """Record an API call"""
        import time
        self.calls.append(time.time())

    def wait_if_needed(self):
        """Wait if we're approaching rate limit"""
        import time
        while not self.can_make_call():
            print("⏳ Rate limit approaching, waiting 5 seconds...")
            time.sleep(5)

# ============================================================================
# CACHING CONFIGURATION
# ============================================================================

CACHE_CONFIG = {
    "enabled": True,
    "cache_dir": "./data/cache",
    "ttl_seconds": 3600,  # 1 hour
    "max_cache_size_mb": 100,
}

# ============================================================================
# COST ESTIMATION
# ============================================================================

def estimate_cost_per_query(config):
    """Estimate API calls per query"""
    calls = 1  # Base answer generation

    if config.get("enable_query_rewriting"):
        calls += 1  # Query rewriting

    if config.get("use_hyde"):
        calls += 1  # HyDE generation

    return calls

def estimate_daily_quota_usage(queries_per_day, config):
    """Estimate daily API calls"""
    calls_per_query = estimate_cost_per_query(config)
    total_calls = queries_per_day * calls_per_query

    return {
        "total_calls": total_calls,
        "calls_per_query": calls_per_query,
        "queries_per_day": queries_per_day,
        "percentage_of_free_tier": (total_calls / 1500) * 100,
        "will_exceed_free_tier": total_calls > 1500,
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_active_config():
    """Print the active configuration"""
    print(f"\n{'='*60}")
    print(f"ACTIVE PROFILE: {ACTIVE_PROFILE}")
    print(f"{'='*60}")
    print(f"Model: {ACTIVE_CONFIG.get('gemini_model')}")
    print(f"Query Rewriting: {ACTIVE_CONFIG.get('enable_query_rewriting')}")
    print(f"Reranking: {ACTIVE_CONFIG.get('enable_reranking')}")
    print(f"Clustering: {ACTIVE_CONFIG.get('enable_clustering')}")
    print(f"HyDE: {ACTIVE_CONFIG.get('use_hyde')}")
    print(f"\nEstimated API calls per query: {estimate_cost_per_query(ACTIVE_CONFIG)}")
    print(f"{'='*60}\n")

def get_optimized_chatbot():
    """Get a chatbot instance with active configuration"""
    import os
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent / "src"))

    from rag_chatbot import RAGChatbot

    chatbot = RAGChatbot(
        api_key=os.getenv("GOOGLE_API_KEY"),
        gemini_model=ACTIVE_CONFIG.get("gemini_model", "gemini-pro"),
        enable_query_rewriting=ACTIVE_CONFIG.get("enable_query_rewriting", False),
        enable_reranking=ACTIVE_CONFIG.get("enable_reranking", True),
        enable_clustering=ACTIVE_CONFIG.get("enable_clustering", False),
    )

    return chatbot

if __name__ == "__main__":
    print_active_config()

    # Example usage estimation
    print("\n📊 Usage Estimation:")
    print(f"{'='*60}")

    for queries in [10, 50, 100, 500]:
        estimate = estimate_daily_quota_usage(queries, ACTIVE_CONFIG)
        print(f"\n{queries} queries/day:")
        print(f"  Total API calls: {estimate['total_calls']}")
        print(f"  % of free tier: {estimate['percentage_of_free_tier']:.1f}%")
        print(f"  Will exceed: {'❌ YES' if estimate['will_exceed_free_tier'] else '✅ NO'}")
