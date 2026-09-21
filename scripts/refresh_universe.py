from src.data.paypay_collector import build_snapshot

if __name__=="__main__":
    snap=build_snapshot("data/universe/latest.json")
    print(f"paypay-universe: {snap['record_count']}")
