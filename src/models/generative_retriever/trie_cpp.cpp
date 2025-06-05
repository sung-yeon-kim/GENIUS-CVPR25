#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <memory>

namespace py = pybind11;

class TrieNode {
public:
    std::unordered_map<int, std::shared_ptr<TrieNode>> children;
};

class Trie {
public:
    Trie(const std::vector<std::vector<int>>& sequences = {}) : root(std::make_shared<TrieNode>()), len(0) {
        for (const auto& seq : sequences) {
            add(seq);
        }
    }

    void append(std::shared_ptr<Trie> trie, int bos_token_id) {
        append_trie = trie;
        this->bos_token_id = bos_token_id;
    }

    void add(const std::vector<int>& sequence) {
        auto node = root;
        for (int token : sequence) {
            if (!node->children.count(token)) {
                node->children[token] = std::make_shared<TrieNode>();
            }
            node = node->children[token];
        }
        ++len;
    }

    std::vector<int> get(const std::vector<int>& prefix_sequence) const {
        return _get_from_trie(prefix_sequence, root, append_trie, bos_token_id);
    }

    std::unordered_map<int, py::object> to_dict() const {
        return _to_dict(root);
    }

    static std::shared_ptr<Trie> from_dict(const std::unordered_map<int, py::object>& d) {
        auto trie = std::make_shared<Trie>();
        _build_from_dict(trie->root, d);
        trie->len = _count_leaves(trie->root);
        return trie;
    }

    int size() const { return len; }

private:
    std::shared_ptr<TrieNode> root;
    std::shared_ptr<Trie> append_trie = nullptr;
    int bos_token_id = -1;
    int len;

    std::vector<int> _get_keys(const std::shared_ptr<TrieNode>& node) const {
        std::vector<int> keys;
        for (const auto& kv : node->children) {
            keys.push_back(kv.first);
        }
        return keys;
    }

    static std::vector<int> _get_from_trie(
        const std::vector<int>& prefix,
        std::shared_ptr<TrieNode> node,
        std::shared_ptr<Trie> append_trie,
        int bos_token_id
    ) {
        if (prefix.empty()) {
            std::vector<int> out;
            for (const auto& kv : node->children) {
                if (kv.first != bos_token_id) {
                    out.push_back(kv.first);
                }
            }
            if (append_trie && node->children.count(bos_token_id)) {
                auto extra = append_trie->_get_keys(append_trie->root);
                out.insert(out.end(), extra.begin(), extra.end());
            }
            return out;
        }

        int head = prefix[0];
        auto it = node->children.find(head);
        if (it != node->children.end()) {
            std::vector<int> rest(prefix.begin() + 1, prefix.end());
            return _get_from_trie(rest, it->second, append_trie, bos_token_id);
        } else if (append_trie) {
            return append_trie->get(prefix);
        }
        return {};
    }

    static std::unordered_map<int, py::object> _to_dict(std::shared_ptr<TrieNode> node) {
        std::unordered_map<int, py::object> result;
        for (const auto& kv : node->children) {
            result[kv.first] = py::cast(_to_dict(kv.second));
        }
        return result;
    }

    static void _build_from_dict(std::shared_ptr<TrieNode> node, const std::unordered_map<int, py::object>& d) {
        for (const auto& kv : d) {
            int key = kv.first;
            auto sub_dict = kv.second.cast<std::unordered_map<int, py::object>>();
            node->children[key] = std::make_shared<TrieNode>();
            _build_from_dict(node->children[key], sub_dict);
        }
    }

    static int _count_leaves(const std::shared_ptr<TrieNode>& node) {
        if (node->children.empty()) return 1;
        int count = 0;
        for (const auto& kv : node->children) {
            count += _count_leaves(kv.second);
        }
        return count;
    }
};

PYBIND11_MODULE(trie_cpp, m) {
    py::class_<Trie, std::shared_ptr<Trie>>(m, "Trie")
        .def(py::init<const std::vector<std::vector<int>>&>())
        .def("append", &Trie::append)
        .def("add", &Trie::add)
        .def("get", &Trie::get)
        .def("to_dict", &Trie::to_dict)
        .def_static("from_dict", &Trie::from_dict)
        .def("size", &Trie::size);
}
