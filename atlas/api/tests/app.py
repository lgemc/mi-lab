"""Every route answers, and the bounds a public server needs are enforced"""

import unittest

from fastapi.testclient import TestClient

from atlas.app import app

client = TestClient(app)

FD = {"nodes": [{"id": "X"}, {"id": "M"}, {"id": "Y"}, {"id": "U", "latent": True}],
      "edges": [["U", "X"], ["U", "Y"], ["X", "M"], ["M", "Y"]]}


class TestRoutes(unittest.TestCase):
    def test_every_route(self):
        calls = [
            ("get", "/api/health", None),
            ("post", "/api/causal/analyze", {**FD, "x": "X", "y": "Y", "given": []}),
            ("post", "/api/causal/rule", {**FD, "rule": 2, "y": ["M"], "z": ["X"]}),
            ("post", "/api/causal/frontdoor", {"pu": 0.5, "px_u": [0.2, 0.8], "pm_x": [0.1, 0.9],
                                               "py_mu": [[0.1, 0.5], [0.3, 0.8]]}),
            ("get", "/api/mlp/meta", None),
            ("post", "/api/mlp/run", {"x": [1, 1, 0], "set": {"n1": 0}}),
            ("post", "/api/mlp/patch", {"clean": [1, 1, 0], "corrupt": [1, 0, 0]}),
            ("post", "/api/mlp/mediation", {"treatment": "b", "mediator": "n1", "base": [1, 0, 0]}),
            ("get", "/api/ioi/meta", None),
            ("post", "/api/ioi/run", {"io": "Mary", "s": "John", "ablate": {"L2.0": "zero"}}),
            ("post", "/api/ioi/patch", {"io": "Mary", "s": "John"}),
            ("post", "/api/ioi/edges", {"io": "Mary", "s": "John", "tau": 0.1}),
            ("get", "/api/lens/meta", None),
            ("post", "/api/lens/probe", {"subject": "Curie", "patch_country": "Japan"}),
            ("post", "/api/superposition", {"sparsity": 0.8}),
            ("post", "/api/sae", {"l1": 0.3}),
            ("post", "/api/sae/sweep", {"mode": "topk"}),
            ("post", "/api/transcoder", {}),
            ("post", "/api/transcoder/graph", {"features": [1, 1, 0, 0, 0]}),
            ("post", "/api/crosscoder", {"mode": "batchtopk"}),
            ("post", "/api/probe", {}),
        ]
        for method, url, body in calls:
            r = getattr(client, method)(url, json=body) if body is not None else getattr(client, method)(url)
            self.assertEqual(r.status_code, 200, (url, r.text[:300]))

    def test_bounds_are_enforced(self):
        self.assertEqual(client.post("/api/sae", json={"latents": 10_000}).status_code, 422)
        self.assertEqual(client.post("/api/ioi/run", json={"io": "Mary", "s": "Mary"}).status_code, 422)
        self.assertEqual(client.post("/api/ioi/run", json={"io": "Eve", "s": "Mary"}).status_code, 422)
        cyc = {"nodes": [{"id": "A"}, {"id": "B"}], "edges": [["A", "B"], ["B", "A"]], "x": "A", "y": "B"}
        self.assertEqual(client.post("/api/causal/analyze", json=cyc).status_code, 422)
        self.assertEqual(client.post("/api/lens/probe", json={"subject": "Curie", "two_hop": 0.5}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
