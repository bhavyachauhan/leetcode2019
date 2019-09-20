import java.util.LinkedList;
import java.util.Queue;

@SuppressWarnings({"WeakerAccess", "unused"})
public class TrieST<Value> {

    public static int R = 256; // Radix (base) - Number of chars in ASCII table
    public TrieNode root;
    private int N;

    static class TrieNode {
        Object value;
        TrieNode[] next = new TrieNode[R];
    }

    public void put(String key, Value val) {
        root = put(root, key, val, 0);
    }

    private TrieNode put(TrieNode node, String key, Value val, int d) {
        if (node == null) {
            node = new TrieNode();
        }

        if (d == key.length()) {
            N += node.value == null ? 1 : 0;
            node.value = val;
            return node;
        }

        char c = key.charAt(d);
        node.next[c] = put(node.next[c], key, val, d + 1);
        return node;
    }

    public Value get(String key) {
        TrieNode node = get(root, key, 0);
        if (node == null) {
            return null;
        }
        //noinspection unchecked
        return (Value) node.value;
    }

    public TrieNode get(TrieNode node, String key, int d) {
        if (node == null) {
            return null;
        }

        if (d == key.length()) {
            return node;
        }

        char c = key.charAt(d);
        return get(node.next[c], key, d + 1);
    }

    public int size() {
        return N;
    }

    public int lazysize() {
        return size(root);
    }

    private int size(TrieNode node) {
        if (node == null) {
            return 0;
        }

        int size = 0;
        if (node.value != null) {
            size++;
        }
        for (char c = 0; c < R; c++) {
            size += size(node.next[c]);
        }
        return size;
    }

    public Iterable<String> keysWithPrefix(String prefix) {
        Queue<String> queue = new LinkedList<>();
        collect(get(root, prefix, 0), prefix, queue);
        return queue;
    }

    public Iterable<String> keys() {
        return keysWithPrefix("");
    }

    private void collect(TrieNode node, String prefix, Queue<String> queue) {
        if (node == null) {
            return;
        }

        if (node.value != null) {
            queue.offer(prefix);
        }

        for (char c = 0; c < R; c++) {
            collect(node.next[c], prefix + c, queue);
        }
    }

    public String lcp() {
        return dfs(root);
    }

    public String dfs(TrieNode node) {
        String str = node.value == null ? "" : (String) node.value;

        for (char c = 0; c < R; c++) {
            if (node.next[c] != null && node.next[c].value != null) {
                String temp = dfs(node.next[c]);
                if (temp.length() > str.length()) {
                    str = temp;
                }
            }
        }


        return str;
    }

}
