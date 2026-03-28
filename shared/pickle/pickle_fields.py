import pickle
from dataclasses import fields


def find_unpickable_fields(obj):
    bad_fields = []
    
    for f in fields(obj):
        value = getattr(obj, f.name)
        try:
            pickle.dumps(value)
        except Exception as e:
            print(f"[FAIL] Field '{f.name}' is not picklable: {type(value)} -> {e}")
            bad_fields.append(f.name)
        else:
            print(f"[OK]   Field '{f.name}' is picklable")
    
    return bad_fields

