"""
风水数据集爬虫 - 配置文件
走 Clash 代理访问 Bing / Houzz / Flickr / Dezeen
"""

TARGET_PER_CLASS  = 1000
MIN_SHORT_SIDE    = 512
MAX_WHITE_RATIO   = 0.72
DOWNLOAD_WORKERS  = 12
OUTPUT_DIR        = "FengShui_Dataset"
REQUEST_PROXIES   = None

CATEGORIES = {

    "main_door": {
        "idx": 0,
        "queries": [
            "front door interior entryway home",
            "home main entrance door inside view",
            "foyer front door hallway interior design",
            "apartment main door inside living",
            "luxury front door entryway interior",
            "front door welcome mat hallway home",
            "home entrance inside view door open",
            "front door interior modern house design",
        ],
    },

    "room_door": {
        "idx": 1,
        "queries": [
            "interior bedroom door hallway home",
            "sliding barn door bedroom interior",
            "french door glass panel room interior",
            "open room door hallway apartment",
            "interior door modern home corridor",
            "pocket door interior living space",
            "bedroom door open inside hallway",
            "wooden interior door home design",
        ],
    },

    "floor_window": {
        "idx": 2,
        "queries": [
            "floor to ceiling window living room",
            "panoramic window modern apartment interior",
            "full height glass window living room sofa",
            "large floor window bedroom interior",
            "floor to ceiling window city view",
            "wall of windows living room modern",
            "floor length window bright room interior",
            "tall glass window open living space",
        ],
    },

    "normal_window": {
        "idx": 3,
        "queries": [
            "small window kitchen interior home",
            "bedroom window curtain interior cozy",
            "bathroom window interior privacy",
            "window seat reading nook interior",
            "kitchen window over sink home",
            "double hung window bedroom interior",
            "casement window room home interior",
            "small square window interior wall",
        ],
    },

    "stove": {
        "idx": 4,
        "queries": [
            "kitchen stove cooktop interior home cooking",
            "gas stove kitchen real home interior",
            "induction stove modern kitchen interior",
            "kitchen range stove counter home",
            "stove oven kitchen interior design",
            "gas burner cooktop kitchen home",
            "kitchen stove pots cooking real",
            "built-in stove kitchen modern home",
        ],
    },

    "sink": {
        "idx": 5,
        "queries": [
            "kitchen sink counter interior home",
            "bathroom sink vanity interior home",
            "kitchen sink window view interior",
            "undermount sink kitchen interior design",
            "double sink kitchen home modern",
            "bathroom vanity sink mirror interior",
            "farmhouse sink kitchen interior real",
            "kitchen sink faucet counter home",
        ],
    },

    "stairs": {
        "idx": 6,
        "queries": [
            "indoor staircase living room home interior",
            "modern staircase home interior design",
            "wooden staircase house interior real",
            "spiral staircase indoor home",
            "open staircase modern house interior",
            "staircase from door view home",
            "floating staircase modern home interior",
            "staircase hallway home interior design",
        ],
    },

    "water_feature": {
        "idx": 7,
        "queries": [
            "aquarium fish tank living room interior",
            "large fish tank home interior design",
            "reef aquarium built-in wall home",
            "indoor fountain water feature living room",
            "fish tank floor standing home interior",
            "aquarium living room real interior",
            "water wall indoor home modern",
            "fish tank home interior real scene",
        ],
    },

    "broad_leaf_live": {
        "idx": 8,
        "queries": [
            "monstera plant living room interior real",
            "fiddle leaf fig tree home interior",
            "large tropical plant living room corner",
            "bird of paradise plant indoor home",
            "pothos plant broad leaf home interior",
            "rubber tree plant living room",
            "philodendron large leaf indoor room",
            "big green leaf indoor plant home",
        ],
    },

    "sharp_leaf_live": {
        "idx": 9,
        "queries": [
            "cactus indoor home room interior real",
            "snake plant narrow sharp leaf indoor",
            "succulent plant sharp indoor home",
            "aloe vera plant indoor home interior",
            "yucca plant sharp leaf indoor room",
            "agave plant indoor home interior",
            "spiky plant indoor living room home",
            "cactus plant corner room interior",
        ],
    },

    "fake_plant": {
        "idx": 10,
        "queries": [
            "artificial plant living room interior decor",
            "fake monstera plant home staging",
            "faux tropical plant apartment interior",
            "artificial pampas grass tall vase living",
            "fake plant dried home interior decor",
            "artificial tree corner living room",
            "faux palm tree home interior design",
            "fake dried grass floor vase room",
        ],
    },

    "bed": {
        "idx": 11,
        "queries": [
            "bedroom bed interior real home design",
            "master bedroom bed interior modern",
            "bed headboard bedroom interior home",
            "king bed bedroom interior natural light",
            "bed room interior design cozy home",
            "platform bed modern bedroom interior",
            "bed against wall bedroom interior",
            "bedroom bed window daylight interior",
        ],
    },

    "sofa": {
        "idx": 12,
        "queries": [
            "sofa living room interior home real",
            "sectional sofa living room modern interior",
            "sofa couch living room interior design",
            "living room sofa window interior bright",
            "sofa with coffee table living room",
            "L-shaped sofa living room interior",
            "velvet sofa living room interior",
            "sofa against wall living room home",
        ],
    },

    "desk": {
        "idx": 13,
        "queries": [
            "home office desk interior room setup",
            "study desk bedroom interior home real",
            "wooden desk home office interior",
            "desk chair home office modern interior",
            "work desk window interior home office",
            "desk bookshelf home study interior",
            "standing desk home office interior",
            "desk computer home room real interior",
        ],
    },

    "dining_table": {
        "idx": 14,
        "queries": [
            "dining table room interior home real",
            "dining room table chairs interior",
            "round dining table interior home modern",
            "dining table window interior home",
            "family dining table room interior",
            "wooden dining table chairs room",
            "dining area table interior home",
            "dining table pendant light interior",
        ],
    },

    "coffee_table": {
        "idx": 15,
        "queries": [
            "coffee table living room interior home",
            "round coffee table sofa interior",
            "glass coffee table living room interior",
            "wooden coffee table sofa interior",
            "coffee table books living room real",
            "marble coffee table living room",
            "low coffee table living room home",
            "coffee table tray living room decor",
        ],
    },

    "mirror": {
        "idx": 16,
        "queries": [
            "large mirror living room interior home",
            "bedroom wall mirror interior home",
            "full length mirror bedroom interior real",
            "decorative mirror hallway interior home",
            "round mirror wall interior home decor",
            "floor mirror bedroom corner interior",
            "mirror above bathroom vanity interior",
            "arch mirror entryway home interior",
        ],
    },

    "beam": {
        "idx": 17,
        "queries": [
            "exposed ceiling beam living room interior",
            "wooden beam ceiling home interior",
            "ceiling beam bedroom interior real",
            "rustic beam ceiling interior home",
            "structural beam ceiling open plan",
            "beam ceiling kitchen interior home",
            "ceiling beam above bed bedroom",
            "industrial beam ceiling loft interior",
        ],
    },

    "toilet": {
        "idx": 18,
        "queries": [
            "toilet bathroom interior home real",
            "bathroom toilet interior design modern",
            "toilet with sink bathroom interior",
            "freestanding toilet bathroom interior",
            "bathroom interior toilet window home",
            "wall-hung toilet bathroom modern",
            "toilet vanity bathroom interior home",
            "small bathroom toilet interior real",
        ],
    },

    "cabinet": {
        "idx": 19,
        "queries": [
            "bookshelf cabinet living room interior",
            "wardrobe cabinet bedroom interior design",
            "kitchen cabinet interior home modern",
            "built-in cabinet living room interior",
            "storage cabinet hallway interior home",
            "display cabinet living room interior",
            "tall cabinet bedroom interior home",
            "sideboard cabinet dining room interior",
        ],
    },
}
