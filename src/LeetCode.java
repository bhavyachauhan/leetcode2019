import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;

@SuppressWarnings({"WeakerAccess", "unused", "ForLoopReplaceableByForEach", "Convert2Lambda"})
public class LeetCode {

    public static void main(String[] args) {

        LeetCode l = new LeetCode();

        String[] strs = {"flower", "flow", "flight"};

        StringBuilder result = new StringBuilder(strs[0]);

        for (int i = 1; i < strs.length; i++) {
            while(strs[i].indexOf(result.toString()) != 0) {
                result.deleteCharAt(result.length() - 1);
                if (result.length() == 0) {
                    System.out.println("Empty");
                }
            }
        }
        System.out.println(result.toString());


        StringBuilder sb = new StringBuilder(strs[0]);

        for (int i = 1; i < strs.length; i++) {
            int index = 0;
            int length = Math.min(sb.length(), strs[i].length());
            while (index < length && sb.charAt(index) == strs[i].charAt(index)) {
                index++;
            }
            sb.delete(index, sb.length());
            if (sb.length() == 0) {
                break;
            }
        }

        System.out.println(sb.toString());
    }

    public void flatten(TreeNode root) {

        if (root == null) {
            return;
        }

        flatten(root.left);
        flatten(root.right);

        TreeNode temp = root.right;
        root.right = root.left;
        root.left = null;

        while(root.right != null) {
            root = root.right;
        }

        root.right = temp;
    }

    public int trap1(int[] height) {

        int result = 0, index = 0;

        Stack<Integer> stack = new Stack<>();
        while (index < height.length) {
            while (!stack.isEmpty() && height[index] > height[stack.peek()]) {
                int maxHeight = stack.pop();

                if (stack.isEmpty()) {
                    break;
                }

                int count = index - stack.peek() - 1;
                int water = Math.min(height[index], height[stack.peek()]) - height[maxHeight];
                result += (count * water);
            }
            stack.push(index++);
        }

        return result;

    }

    public boolean isValidParenthesis(String s) {

        int len = s.length();
        int newLen = 0;

        while (len != newLen) {
            len = s.length();
            s = s.replaceAll("\\Q()\\E", "");
            s = s.replaceAll("\\Q{}\\E", "");
            s = s.replaceAll("\\Q[]\\E", "");
            newLen = s.length();
        }

        return s.length() == 0;
    }

    public ListNode swapPairs(ListNode head) {

        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode curr = dummy;

        while (curr.next != null && curr.next.next != null) {
            ListNode node1 = curr.next;
            curr.next = node1.next;
            node1.next = node1.next.next;
            curr.next.next = node1;
            curr = curr.next.next;
        }

        return dummy.next;
    }

    public String customSortString(String S, String T) {

        Set<Character> set = new HashSet<>();
        for (char c : S.toCharArray()) {
            set.add(c);
        }

        char[] arr = T.toCharArray();
        int left = 0, right = arr.length - 1;
        while (left < right) {
            if (!set.contains(arr[left])) {
                char t = arr[left];
                arr[left] = arr[right];
                arr[right--] = t;
            } else {
                left++;
            }
        }

        List<Character> list = new ArrayList<>();
        for (char c : arr) {
            list.add(c);
        }

        list.sort(new Comparator<Character>() {

            @Override
            public int compare(Character c1, Character c2) {
                if (S.contains(c1.toString()) && S.contains(c2.toString())) {
                    return S.indexOf(c1) - S.indexOf(c2);
                } else {
                    return 0;
                }
            }

        });

        StringBuilder sb = new StringBuilder();
        for (char c : list) {
            sb.append(c);
        }
        return sb.toString();
    }

    public List<TreeNode> deepestPath(TreeNode root) {
        List<List<TreeNode>> paths = new ArrayList<>();

        List<TreeNode> path = new ArrayList<>();
        path.add(root);
        deepestPath(root, paths, path);

        List<TreeNode> result = new ArrayList<>();
        for (List<TreeNode> p: paths) {
            if (p.size() > result.size()) {
                result = p;
            }
        }
        return result;
    }

    void deepestPath(TreeNode node, List<List<TreeNode>> paths, List<TreeNode> path) {
        if (node.left == null && node.right == null) {
            List<TreeNode> temp = new ArrayList<>(path);
            paths.add(temp);
            System.out.println(temp);
            return;
        }

        if (node.left != null) {
            path.add(node.left);
            deepestPath(node.left, paths, path);
            path.remove(path.size() - 1);
        }

        if (node.right != null) {
            path.add(node.right);
            deepestPath(node.right, paths, path);
            path.remove(path.size() - 1);
        }
    }


    public List<String> binaryTreePaths(TreeNode root) {
        List<String> paths = new ArrayList<>();
        binaryTreePaths(root, paths, "");
        return paths;
    }

    void binaryTreePaths(TreeNode node, List<String> paths, String path) {
        if (node == null) {
            return;
        }

        path += node.val + "->";

        if (node.left == null && node.right == null) {
            path = path.substring(0, path.length() - 2);
            paths.add(path);
            return;
        }

        binaryTreePaths(node.left, paths, path);
        binaryTreePaths(node.right, paths, path);
    }

    public void merge1(int[] nums1, int m, int[] nums2, int n) {

        int i = 0, j = 0;

        while (i < m && j < n) {
            if (nums2[j] < nums1[i]) {
                int temp = nums1[i];
                nums1[i] = nums2[j];
                nums2[j] = temp;
                j++;
            } else {
                i++;
            }

        }


    }

    public boolean isAlienSorted(String[] words, String order) {

        List<String> wordsList = new ArrayList<>(Arrays.asList(words));

        wordsList.sort(new Comparator<String>() {

            public int compare(String s1, String s2) {
                int l = 0, r = 0;
                while (l < s1.length() && r < s2.length() && s1.charAt(l) == s2.charAt(r)) {
                    l++;
                    r++;
                }

                if (l == s1.length() && r == s2.length()) {
                    return 0;
                }

                if (l == s1.length()) {
                    return -1;
                }

                if (r == s2.length()) {
                    return 1;
                }

                return order.indexOf(s1.charAt(l)) - order.indexOf(s2.charAt(r));
            }

        });

        for (int i = 0; i < words.length; i++) {
            if (!words[i].equals(wordsList.get(i))) {
                return false;
            }
        }

        return true;

    }


    static class AirBNBQueue {

        int ARR_CAPACITY = 5;

        int capacity = 1;
        String[][] elems;
        int count = 0;

        public AirBNBQueue() {
            elems = new String[capacity][ARR_CAPACITY];
        }

        public void enqueue(String value) {
            int row = count / ARR_CAPACITY;
            int col = count % ARR_CAPACITY;
            count++;

            if (row >= capacity) {
                expandCapacity();
            }

            elems[row][col] = value;
        }

        public String dequeue() {

            if (count == 0) {
                return null;
            }

            String elem = elems[0][0];

            String[][] temp = new String[elems.length][ARR_CAPACITY];
            copyArr(elems, temp);

            int c = count;
            count = 0;

            for (int i = 0; i < temp.length; i++) {
                for (int j = 0; j < temp[0].length; j++) {
                    if (count == c - 1) {
                        break;
                    }
                    if (i != 0 || j != 0) {
                        enqueue(temp[i][j]);
                    }
                }
            }

            return elem;
        }

        public int size() {
            return count;
        }

        private void expandCapacity() {

            String[][] temp = new String[capacity][ARR_CAPACITY];
            copyArr(elems, temp);

            capacity = 2 * capacity;
            elems = new String[capacity][ARR_CAPACITY];
            copyArr(temp, elems);

        }

        private void copyArr(String[][] srcArr, String[][] destArr) {
            for (int i = 0; i < srcArr.length; i++) {
                String[] arr = srcArr[i];
                String[] copy = new String[ARR_CAPACITY];
                System.arraycopy(arr, 0, copy, 0, arr.length);
                destArr[i] = copy;
            }
        }

    }

    int sign(int divisor, int dividend) {
        return divisor < 0 ^ dividend < 0 ? -1 : 1;
    }

    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n <= 0) {
            return Collections.emptyList();
        }

        if (n == 1 && edges.length == 0) {
            return Collections.singletonList(0);
        }

        List<HashSet<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            adj.add(new HashSet<>());
        }

        for (int[] edge : edges) {
            adj.get(edge[0]).add(edge[1]);
            adj.get(edge[1]).add(edge[0]);
        }

        List<Integer> leaves = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (adj.get(i).size() == 1) {
                leaves.add(i);
            }
        }

        while (n > 2) {
            List<Integer> nextLeaves = new ArrayList<>();
            for (int leaf : leaves) {
                n--;
                int neighbor = adj.get(leaf).iterator().next();
                adj.get(neighbor).remove(leaf);

                if (adj.get(neighbor).size() == 1) {
                    nextLeaves.add(neighbor);
                }
            }
            leaves = nextLeaves;
        }

        return leaves;
    }

    public int rob(int[] nums) {
        int prev = 0;
        int curr = 0;

        for (int i : nums) {
            int temp = curr;
            curr = Math.max(prev + i, curr);
            prev = temp;
        }

        return curr;
    }

    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            n = n & n - 1;
            count++;
        }
        return count;
    }

    public int getSum(int a, int b) {

        while (b != 0) {
            int carry = a & b;
            a = a ^ b;
            b = carry << 1;
        }

        return a;
    }

    public int subarraySum(int[] nums, int k) {
        int count = 0, sum = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);

        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (map.containsKey(sum - k)) {
                count += map.get(sum - k);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }

        return count;
    }

    public List<String> generateParenthesis(int n) {
        final List<String> output = new LinkedList<>();
        final StringBuilder sb = new StringBuilder();
        generateParenthesis(n, n, output, sb);
        return output;
    }

    private void generateParenthesis(final int rob, final int rcb, final List<String> output, final StringBuilder sb) {
        if (rob == 0 && rcb == 0) {
            output.add(sb.toString());
            return;
        }

        if (rcb > rob) {
            sb.append(")");
            generateParenthesis(rob, rcb - 1, output, sb);
            sb.deleteCharAt(sb.length() - 1);
        }

        if (rob > 0) {
            sb.append("(");
            generateParenthesis(rob - 1, rcb, output, sb);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    final Object lock = new Object();

    @SuppressWarnings("AnonymousHasLambdaAlternative")
    boolean threading() {
        sb = new StringBuffer();
        isDone1 = false;
        isDone2 = false;

        Thread t1 = new Thread() {

            @Override
            public void run() {
                System.out.println("Calling func1");
                func1();
            }

        };


        Thread t2 = new Thread() {
            @Override
            public void run() {
                System.out.println("Calling func2");
                try {
                    func2();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        };

        Thread t3 = new Thread() {
            @Override
            public void run() {
                System.out.println("Calling func3");
                try {
                    func3();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        };

        t3.start();
        t1.start();
        t2.start();

        synchronized (lock) {
            try {
                lock.wait();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        System.out.println(sb.toString());
        return sb.toString().equals("FirstSecondThird");
    }

    StringBuffer sb;

    boolean isDone1, isDone2;

    final Object lock1 = new Object();
    final Object lock2 = new Object();

    public void func1() {
        sb.append("First");
        isDone1 = true;
        synchronized (lock1) {
            lock1.notify();
        }
    }

    public void func2() throws InterruptedException {
        if (!isDone1) {
            synchronized (lock1) {
                lock1.wait();
            }
        }
        sb.append("Second");
        isDone2 = true;
        synchronized (lock2) {
            lock2.notify();
        }
    }

    public void func3() throws InterruptedException {
        if (!isDone2) {
            synchronized (lock2) {
                lock2.wait();
            }
        }
        sb.append("Third");
        synchronized (lock) {
            lock.notify();
        }
    }

    public String reverseStr(String s, int k) {

        StringBuilder sb = new StringBuilder();
        char[] arr = s.toCharArray();
        int i = 0;
        int j;
        while (i < arr.length) {
            if (i % (2 * k) == 0) {
                j = Math.min(i + k - 1, arr.length - 1);
                while (j >= i) {
                    sb.append(arr[j--]);
                }
                i += k;
            } else {
                sb.append(arr[i++]);
            }
        }

        return sb.toString();
    }

    public List<Boolean> prefixesDivBy5(int[] A) {

        long num = 0;
        long n = 0;
        List<Boolean> list = new ArrayList<>();
        List<Long> nums = new ArrayList<>();
        for (int i = 0; i < A.length; i++) {
            n = (n * 2 + A[i]);
            nums.add(n);
            num = (num * 2 + A[i]) % 5;
            list.add(num == 0);
        }
        System.out.println(nums);
        return list;

    }

    public int[][] intervalIntersection(int[][] A, int[][] B) {

        int p1 = 0;
        int p2 = 0;

        List<int[]> result = new ArrayList<>();

        while (p1 < A.length && p2 < B.length) {
            int[] i1 = A[p1];
            int[] i2 = B[p2];

            int[] intersection = getIntersection(i1, i2);
            if (intersection == null) {
                if (i1[1] > i2[1]) {
                    p2++;
                } else if (i2[1] > i1[1]) {
                    p1++;
                }
                continue;
            }

            result.add(intersection);

            if (intersection[1] == i1[1]) {
                p1++;
            }
            if (intersection[1] == i2[1]) {
                p2++;
            }
        }

        return result.toArray(new int[0][]);

    }

    int[] getIntersection(int[] i1, int[] i2) {
        int[] intersection = new int[2];
        if (i1[0] >= i2[0] && i1[0] <= i2[1]) {
            intersection[0] = i1[0];
        } else if (i2[0] >= i1[0] && i2[0] <= i1[1]) {
            intersection[0] = i2[0];
        } else {
            return null;
        }

        if (i1[1] >= i2[0] && i1[1] <= i2[1]) {
            intersection[1] = i1[1];
        } else if (i2[1] >= i1[0] && i2[1] <= i1[1]) {
            intersection[1] = i2[1];
        }

        return intersection;
    }

    public boolean isValidBST(TreeNode node) {
        return isValidBST(node, null, null);
    }

    public boolean isValidBST(TreeNode node, Integer lower, Integer upper) {
        if (node == null) {
            return true;
        }

        boolean isValid = (lower == null || node.val > lower) &&
            (upper == null || node.val < upper);

        return isValid &&
            isValidBST(node.left, lower, node.val) &&
            isValidBST(node.right, node.val, upper);
    }

    public String minWindow(String s, String t) {

        int left = 0, right;
        char[] arr = s.toCharArray();
        String window = "";

        while (left < arr.length) {
            Map<Character, Integer> map = new HashMap<>();
            for (char c : t.toCharArray()) {
                map.put(c, map.getOrDefault(c, 0) + 1);
            }

            while (left < arr.length && !map.containsKey(arr[left])) {
                left++;
            }

            right = left;

            while (right < arr.length && !map.isEmpty()) {
                if (map.containsKey(arr[right])) {
                    int num = map.get(arr[right]) - 1;
                    if (num == 0) {
                        map.remove(arr[right]);
                    } else {
                        map.put(arr[right], num);
                    }
                }
                right++;
            }

            if (map.isEmpty()) {
                String newWindow = s.substring(left, right);
                window = window.isEmpty() || window.length() > newWindow.length()
                    ? newWindow : window;
            }
            left++;
        }

        return window;
    }

    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }

        int i = nums.length - 2;

        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }

        if (i < 0) {
            Arrays.sort(nums);
            return;
        }

        int index = i + 1;
        while (index < nums.length && nums[index] > nums[i]) {
            index++;
        }

        int temp = nums[i];
        nums[i] = nums[index - 1];
        nums[index - 1] = temp;
        Arrays.sort(nums, i + 1, nums.length);
    }

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {

        int l = beginWord.length();

        Map<String, List<String>> combos = new HashMap<>();
        for (String word : wordList) {
            for (int i = 0; i < l; i++) {
                String newWord = word.substring(0, i) + '*' + word.substring(i + 1, l);
                List<String> comboWords = combos.getOrDefault(newWord, new ArrayList<>());
                comboWords.add(word);
                combos.put(newWord, comboWords);
            }
        }

        return 0;

    }

    static class Node {
        public int val;
        public Node next;
        public Node random;

        public Node() {}

        public Node(int _val, Node _next, Node _random) {
            val = _val;
            next = _next;
            random = _random;
        }
    }

    public Node copyRandomList(Node head) {
        Map<Node, Node> nodes = new HashMap<>();
        Node curr = head;
        while (curr != null && !nodes.containsKey(curr)) {
            Node nextNode = curr.next;
            nodes.put(curr, new Node(curr.val, null, null));
            curr = curr.next;
        }

        curr = head;
        while (curr != null) {
            nodes.get(curr).next = nodes.get(curr.next);
            nodes.get(curr).random = nodes.get(curr.random);

            curr = curr.next;
        }

        return nodes.get(head);
    }

    int nr, nc;

    public List<String> findWords(char[][] board, String[] words) {
        if (board.length == 0 || board[0].length == 0) {
            return new ArrayList<>();
        }

        nr = board.length;
        nc = board[0].length;

        List<String> result = new ArrayList<>();
        for (String word : words) {
            if (hasWord(board, word)) {
                result.add(word);
            }
        }

        return result;
    }

    public boolean hasWord(char[][] board, String word) {

        boolean[][] visited = new boolean[nr][nc];
        int wordIndex = 0;

        for (int r = 0; r < nr; r++) {
            for (int c = 0; c < nc; c++) {
                if (board[r][c] == word.charAt(wordIndex) && !visited[r][c]) {
                    if (dfs(board, word, wordIndex, r, c, visited)) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    boolean isValid(char[][] board, String word, int wordIndex, int r, int c, boolean[][] visited) {
        return r >= 0 && r < nr
            && c >= 0 && c < nc
            && !visited[r][c] && board[r][c] == word.charAt(wordIndex);
    }

    boolean dfs(char[][] board, String word, int wordIndex, int r, int c, boolean[][] visited) {
        if (wordIndex == word.length()) {
            return true;
        }

        if (isValid(board, word, wordIndex, r, c, visited)) {
            visited[r][c] = true;

            return dfs(board, word, wordIndex + 1, r + 1, c, visited) ||
                dfs(board, word, wordIndex + 1, r - 1, c, visited) ||
                dfs(board, word, wordIndex + 1, r, c + 1, visited) ||
                dfs(board, word, wordIndex + 1, r, c - 1, visited);
        }
        return false;
    }


    public double findMedianSortedArrays(int[] nums1, int[] nums2) {

        int n = nums1.length + nums2.length;
        int mid = n / 2;

        int[] merged = new int[nums1.length + nums2.length];
        int i1 = 0, i2 = 0;
        int index = 0;

        while (index <= n / 2 && i1 < nums1.length && i2 < nums2.length) {
            if (nums1[i1] < nums2[i2]) {
                merged[index++] = nums1[i1++];
            } else {
                merged[index++] = nums2[i2++];
            }
        }

        while (index <= n / 2 && i1 < nums1.length) {
            merged[index++] = nums1[i1++];
        }

        while (index <= n / 2 && i2 < nums2.length) {
            merged[index++] = nums2[i2++];
        }

        if (n % 2 == 0) {
            return (double) (merged[mid] + merged[mid - 1]) / 2;
        } else {
            return (double) merged[mid];
        }

    }

    public int move(int[][] board, int row, int col, int player) {
        board[row][col] *= (player + 1);
        if (wins(board, player)) {
            return player;
        }
        return 0;
    }

    boolean wins(int[][] board, int player) {
        int n = board.length;
        int total = (player + 1) * n * ((n * n) + 1) / 2;

        for (int i = 0; i < n; i++) {
            int r = 0;
            int c = 0;
            int d = 0;
            int rd = 0;
            for (int j = 0; j < n; j++) {
                r += board[i][j];
                c += board[j][i];
                d += board[j][j];
                rd += board[j][n - j - 1];
            }
            if (r == total || c == total || d == total || rd == total) {
                return true;
            }
        }

        return false;
    }

    boolean verifyMagicSquare(int[][] board) {
        int n = board.length;
        int total = n * ((n * n) + 1) / 2;

        for (int i = 0; i < n; i++) {
            int r = 0;
            int c = 0;
            int d = 0;
            int rd = 0;
            for (int j = 0; j < n; j++) {
                r += board[i][j];
                c += board[j][i];
                d += board[j][j];
                rd += board[j][n - j - 1];
            }
            if (r != total || c != total || d != total || rd != total) {
                return false;
            }
        }

        return true;
    }

    int[][] createMagicSquare(int n) {
        int[][] board = new int[n][n];
        int i = n / 2, j = n - 1;
        int num = 1;

        while (num <= n * n) {
            if (i == -1 && j == n) { //Rule 3
                i = 0;
                j = n - 2;
            } else { //Rule 1
                j = j >= n ? 0 : j;
                i = i < 0 ? n - 1 : i;
            }

            if (board[i][j] == 0) {
                board[i--][j++] = num++;
            } else { //Rule 2
                i += 1;
                j -= 2;
            }
        }
        return board;
    }

    public int[][] kClosest(int[][] points, int K) {

        int[][] result = new int[K][2];

        List<int[]> sortedPoints = new ArrayList<>(Arrays.asList(points));

        sortedPoints.sort(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return getDistanceSquare(o1) - getDistanceSquare(o2);
            }
        });

        for (int i = 0; i < K; i++) {
            result[i] = sortedPoints.get(0);
        }

        return result;
    }

    int getDistanceSquare(int[] point) {
        return (point[0] * point[0]) + (point[1] * point[1]);
    }

    public List<Integer> partitionLabels(String S) {
        int[] last = new int[26];
        for (int i = 0; i < S.length(); i++) {
            last[S.charAt(i) - 'a'] = i;
        }


        int anchor = 0, index = 0;
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < S.length(); i++) {
            int lastI = last[S.charAt(i) - 'a'];
            index = Math.max(index, lastI);
            if (i == index) {
                result.add(i - anchor + 1);
                anchor = i + 1;
            }
        }

        return result;
    }

    class MinStack {

        int size = 16, count;
        int[] elems;
        PriorityQueue<Integer> sortedElems;

        /**
         * initialize your data structure here.
         */
        public MinStack() {
            elems = new int[size];
            count = 0;
            sortedElems = new PriorityQueue<>();
        }

        public void push(int x) {
            sortedElems.offer(x);
            expandIfNeeded();
            elems[count++] = x;
        }

        void expandIfNeeded() {
            if (count + 1 >= size) {
                size *= 2;
                int[] temp = new int[elems.length];
                System.arraycopy(elems, 0, temp, 0, elems.length);
                elems = new int[size];
                System.arraycopy(temp, 0, elems, 0, temp.length);
            }
        }

        public void pop() {
            sortedElems.remove(elems[--count]);
        }

        public int top() {
            return elems[count - 1];
        }

        public int getMin() {
            if (sortedElems.isEmpty()) {
                throw new IndexOutOfBoundsException();
            }
            return sortedElems.peek();
        }
    }

    public String mostCommonWord(String paragraph, String[] banned) {
        String[] words = paragraph.toLowerCase().replaceAll("[^a-z0-9 ]", " ").split("\\s+");
        Set<String> bannedWords = new HashSet<>(Arrays.asList(banned));

        Map<String, Integer> map = new HashMap<>();

        String result = null;
        int maxCount = 0;

        for (String word : words) {
            word = word.trim();
            if (bannedWords.contains(word)) {
                continue;
            }
            int count = map.getOrDefault(word, 0) + 1;
            if (count >= maxCount) {
                result = word;
                maxCount = count;
            }
            map.put(word, count);
        }

        return result;

    }


    public int islandPerimeter(int[][] grid) {

        if (grid.length == 0 || grid[0].length == 0) {
            return 0;
        }

        boolean[][] visited = new boolean[grid.length][grid[0].length];

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1 && !visited[i][j]) {
                    return dfs(grid, i, j, visited);
                }
            }
        }

        return 0;
    }


    int dfs(int[][] grid, int r, int c, boolean[][] visited) {

        if (r >= 0 && r < grid.length && c >= 0 && c < grid[0].length && grid[r][c] == 1) {
            if (visited[r][c]) {
                return 0;
            }

            visited[r][c] = true;
            return dfs(grid, r + 1, c, visited) +
                dfs(grid, r - 1, c, visited) +
                dfs(grid, r, c + 1, visited) +
                dfs(grid, r, c - 1, visited);
        } else {
            return 1;
        }

    }

    public String[] reorderLogFiles(String[] logs) {

        List<String> wordLogs = new ArrayList<>();
        List<String> digitLogs = new ArrayList<>();

        for (int i = 0; i < logs.length; i++) {
            String log = logs[i];

            String firstLogWord = log.split(" ")[1];
            if (Character.isDigit(firstLogWord.charAt(0))) {
                digitLogs.add(log);
            } else {
                wordLogs.add(log);
            }
        }

        wordLogs.sort(new Comparator<String>() {

            @Override
            public int compare(String s1, String s2) {
                String i1 = s1.substring(0, s1.indexOf(" "));
                String l1 = s1.substring(i1.length() + 1);

                String i2 = s2.substring(0, s2.indexOf(" "));
                String l2 = s2.substring(i2.length() + 1);

                if (l1.equals(l2)) {
                    return i1.compareTo(i2);
                } else {
                    return l1.compareTo(l2);
                }
            }

        });

        String[] wordLogsArray = wordLogs.toArray(new String[0]);
        String[] digitLogsArray = digitLogs.toArray(new String[0]);

        System.arraycopy(wordLogsArray, 0, logs, 0, wordLogsArray.length);
        System.arraycopy(digitLogsArray, 0, logs, wordLogsArray.length, digitLogsArray.length);

        return logs;
    }

    public int integerBreak(int n) {
        int[] dp = new int[n + 1];
        dp[1] = 1;

        for (int i = 2; i < n + 1; i++) {
            for (int j = 1; j < i; j++) {
                int t1 = Math.max(dp[i - j], i - j);
                int t = j * t1;
                dp[i] = Math.max(dp[i], t);
            }
        }

        return dp[n];
    }

    public boolean wordBreak(String s, List<String> wordDict) {

        Set<String> set = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;

        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && set.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[s.length()];
    }

    public void merge(int[] nums1, int m, int[] nums2, int n) {

        if (m == 0 && n > 0) {
            System.arraycopy(nums2, 0, nums1, 0, n);
            return;
        }

        int index = m-- + n-- - 1;
        while (n >= 0 && m >= 0 && index >= 0) {
            if (nums1[m] < nums2[n]) {
                nums1[index--] = nums2[n--];
            } else {
                nums1[index--] = nums1[m--];
            }
        }

    }

    public int[][] updateMatrix(int[][] matrix) {

        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == 0) {
                    queue.add(new int[]{i, j});
                } else {
                    matrix[i][j] = Integer.MAX_VALUE;
                }
            }
        }

        int[][] directions = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        while (!queue.isEmpty()) {
            int[] top = queue.remove();
            for (int[] dir : directions) {
                int r = top[0];
                int c = top[1];
                int nr = r + dir[0];
                int nc = c + dir[1];

                if (nr < 0 || nr >= matrix.length || nc < 0 || nc >= matrix[0].length || matrix[nr][nc] <= matrix[r][c] + 1) {
                    continue;
                }

                matrix[nr][nc] = matrix[r][c] + 1;
                queue.add(new int[]{nr, nc});
            }
        }

        return matrix;
    }


    public boolean wordPattern(String pattern, String str) {

        Map<Character, List<Integer>> map = new LinkedHashMap<>();

        for (int i = 0; i < pattern.length(); i++) {
            char c = pattern.charAt(i);
            if (map.containsKey(c)) {
                map.get(c).add(i);
            } else {
                List<Integer> list = new ArrayList<>();
                list.add(i);
                map.put(c, list);
            }
        }

        String[] arr = str.split(" ");
        Map<String, List<Integer>> strMap = new LinkedHashMap<>();

        for (int i = 0; i < arr.length; i++) {
            String s = arr[i];
            if (strMap.containsKey(s)) {
                strMap.get(s).add(i);
            } else {
                List<Integer> list = new ArrayList<>();
                list.add(i);
                strMap.put(s, list);
            }
        }

        if (map.size() != strMap.size()) {
            return false;
        }
        List<List<Integer>> patternList = new ArrayList<>(map.values());
        List<List<Integer>> strList = new ArrayList<>(strMap.values());
        for (int i = 0; i < strMap.size(); i++) {
            if (!patternList.get(i).equals(strList.get(i))) {
                return false;
            }
        }

        return true;
    }

    public int jump(int[] nums) {
        int farthestJump = 0, prevJump = 0, steps = 0;

        for (int i = 0; i < nums.length - 1; i++) {
            farthestJump = Math.max(farthestJump, i + nums[i]);
            if (i == prevJump) {
                steps++;
                prevJump = farthestJump;
                if (farthestJump >= nums.length - 1) {
                    break;
                }
            }
        }

        return steps;
    }

    public int findUnsortedSubarray(int[] nums) {

        int start = Integer.MAX_VALUE, end = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] > nums[i + 1]) {
                start = Math.min(start, i);
                end = Math.max(end, i + 1);
            }
        }

        if (start < end) {
            return end - start + 1;
        }
        return 0;
    }

    public int findLength(int[] A, int[] B) {
        int result = 0;

        int[][] memo = new int[A.length + 1][B.length + 1];
        for (int i = A.length - 1; i >= 0; --i) {
            for (int j = B.length - 1; j >= 0; --j) {
                if (A[i] == B[j]) {
                    memo[i][j] = memo[i + 1][j + 1] + 1;
                    if (result < memo[i][j]) {
                        result = memo[i][j];
                    }
                }
            }
        }

        return result;
    }

    public boolean isSubsequence(String s, String t) {

        for (char c : s.toCharArray()) {
            int index = t.indexOf(c);
            if (index >= 0) {
                t = t.substring(index + 1);
            } else {
                return false;
            }
        }

        return true;

    }

    public List<List<String>> groupAnagrams1(String[] strs) {

        Map<String, Map<Character, Integer>> map = new HashMap<>();
        for (String s : strs) {
            Map<Character, Integer> m = new HashMap<>();
            for (char c : s.toCharArray()) {
                m.put(c, m.getOrDefault(c, 0) + 1);
            }
            map.put(s, m);
        }

        Map<Map<Character, Integer>, List<String>> reverseMap = new HashMap<>();
        for (Map.Entry<String, Map<Character, Integer>> entry : map.entrySet()) {
            if (reverseMap.containsKey(entry.getValue())) {
                reverseMap.get(entry.getValue()).add(entry.getKey());
            } else {
                //noinspection ArraysAsListWithZeroOrOneArgument
                reverseMap.put(entry.getValue(), new ArrayList<>(Arrays.asList(entry.getKey())));
            }
        }

        return new ArrayList<>(reverseMap.values());
    }

    public String longestWord(String[] words) {
        if (words == null || words.length == 0) {
            return "";
        }

        TrieST<String> trie = new TrieST<>();
        for (String s : words) {
            trie.put(s, s);
        }

        return dfs(trie.root);
    }

    public String dfs(TrieST.TrieNode node) {
        String str = node.value == null ? "" : (String) node.value;

        for (char c = 0; c < TrieST.R; c++) {
            if (node.next[c] != null && node.next[c].value != null) {
                String nextStr = dfs(node.next[c]);
                if (nextStr != null && nextStr.length() > str.length()) {
                    str = nextStr;
                }
            }
        }
        return str;
    }

    public List<Interval> employeeFreeTime(List<List<Interval>> schedule) {

        List<Interval> merged = new ArrayList<>();

        for (int i = 0; i < schedule.size(); i++) {
            merged.addAll(schedule.get(i));
        }

        merged.sort(new Comparator<Interval>() {

            public int compare(Interval i1, Interval i2) {
                if (i1.start == i2.start) {
                    return i1.end - i2.end;
                } else {
                    return i1.start - i2.start;
                }
            }

        });

        for (int i = 0; i < merged.size() - 1; i++) {
            Interval i1 = merged.get(i);
            Interval i2 = merged.get(i + 1);

            if (i1.end >= i2.start) {
                merged.set(i, new Interval(i1.start, Math.max(i1.end, i2.end)));
                merged.remove(i + 1);
                i--;
            }

        }

        List<Interval> freeTime = new ArrayList<>();
        for (int i = 0; i < merged.size() - 1; i++) {
            freeTime.add(new Interval(merged.get(i).end, merged.get(i + 1).start));
        }

        return freeTime;
    }

    public List<String> subdomainVisits(String[] cpdomains) {

        Map<String, Integer> map = new HashMap<>();

        for (String s : cpdomains) {
            String[] comp = s.split(" ");
            int num = Integer.valueOf(comp[0]);
            String domain = comp[1];

            while (!domain.isEmpty()) {
                int newNum = 0;
                if (map.containsKey(domain)) {
                    newNum = map.get(domain);
                }
                map.put(domain, num + newNum);
                if (domain.contains(".")) {
                    domain = domain.substring(domain.indexOf('.') + 1);
                } else {
                    domain = "";
                }
            }
        }

        List<String> result = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            result.add(entry.getValue() + " " + entry.getKey());
        }

        return result;
    }

    public List<String> letterCombinations(String digits) {

        char[][] arr = new char[][]{
            "abc".toCharArray(),
            "def".toCharArray(),
            "ghi".toCharArray(),
            "jkl".toCharArray(),
            "mno".toCharArray(),
            "pqrs".toCharArray(),
            "tuv".toCharArray(),
            "wxyz".toCharArray()
        };

        List<String> result = new ArrayList<>();
        char[] input = digits.toCharArray();
        result.add("");
        for (char c : input) {
            result = expand(result, arr[Character.getNumericValue(c) - 2]);
        }

        return result;
    }

    List<String> expand(List<String> list, char[] arr) {
        List<String> newList = new ArrayList<>();
        for (String s : list) {
            for (char c : arr) {
                newList.add(s + c);
            }
        }
        return newList;
    }

    public String longestCommonPrefix1(String[] strs) {

        if (strs == null || strs.length == 0) {
            return "";
        }

        if (strs.length == 1) {
            return strs[0];
        }

        StringBuilder sb = new StringBuilder(strs[0]);

        for (int i = 1; i < strs.length; i++) {
            for (int j = 0; j < sb.length(); j++) {
                if (j >= strs[i].length() || strs[i].charAt(j) != sb.charAt(j)) {
                    sb.delete(j, sb.length());
                }
            }
        }

        return sb.toString();
    }

    public String addBinary(String a, String b) {

        StringBuilder result = new StringBuilder();
        int remainder = 0;

        Stack<Character> sa = new Stack<>();
        for (char c : a.toCharArray()) {
            sa.push(c);
        }

        Stack<Character> sb = new Stack<>();
        for (char c : b.toCharArray()) {
            sb.push(c);
        }

        while (!sa.isEmpty() && !sb.isEmpty()) {
            char ca = sa.pop();
            char cb = sb.pop();

            if (ca == cb) {
                if (ca == '1') {
                    result.append(remainder == 1 ? '1' : '0');
                    remainder = 1;
                } else {
                    result.append(remainder == 0 ? '0' : '1');
                    remainder = 0;
                }
            } else {
                result.append(remainder == 0 ? '1' : '0');
            }
        }

        while (!sa.isEmpty()) {
            char c = sa.pop();
            if (c == '1') {
                result.append(remainder == 0 ? '1' : '0');
            } else {
                result.append(remainder == 0 ? '0' : '1');
                remainder = 0;
            }
        }

        while (!sb.isEmpty()) {
            char c = sb.pop();
            if (c == '1') {
                result.append(remainder == 0 ? '1' : '0');
            } else {
                result.append(remainder == 0 ? '0' : '1');
                remainder = 0;
            }
        }

        if (remainder == 1) {
            result.append('1');
        }

        return result.reverse().toString();
    }

    public int peakIndexInMountainArray(int[] A) {
        return peakIndexMountainArray(A, 0, A.length);
    }

    int peakIndexMountainArray(int[] A, int start, int end) {
        if (start >= end) {
            return -1;
        }

        int mid = (start + end) / 2;
        if (A[mid] > A[mid - 1] && A[mid] > A[mid + 1]) {
            return mid;
        }

        if (mid - 1 >= 0 && A[mid] < A[mid - 1]) {
            end = mid;
        } else if (mid + 1 < A.length && A[mid] < A[mid + 1]) {
            start = mid;
        }

        return peakIndexMountainArray(A, start, end);
    }

    public char findTheDifference(String s, String t) {

        char[] sc = s.toCharArray();
        char[] tc = t.toCharArray();

        Arrays.sort(sc);
        Arrays.sort(tc);

        int i = 0;
        while (i < sc.length && i < tc.length) {
            if (sc[i] != tc[i]) {
                if (sc.length > tc.length) {
                    return sc[i];
                } else {
                    return tc[i];
                }
            }
            i++;
        }

        if (i < sc.length) {
            return sc[i];
        } else {
            return tc[i];
        }

    }

    public int[] singleNumber(int[] nums) {

        Set<Integer> set = new HashSet<>();
        Set<Integer> result = new HashSet<>();

        for (int i : nums) {
            if (set.add(i)) {
                result.add(i);
            } else {
                result.remove(i);
            }
        }

        int[] arr = new int[result.size()];
        Iterator<Integer> it = result.iterator();
        for (int i = 0; i < arr.length; i++) {
            arr[i] = it.next();
        }
        return arr;
    }

    static int WHITE = 0;
    static int GRAY = 1;
    static int BLACK = 2;

    boolean isPossible = true;

    @SuppressWarnings({"ForLoopReplaceableByForEach", "unchecked"})
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        HashSet<Integer>[] adj = new HashSet[numCourses];
        int[] colors = new int[numCourses];

        for (int i = 0; i < adj.length; i++) {
            adj[i] = new HashSet<>();
        }

        for (int i = 0; i < prerequisites.length; i++) {
            adj[prerequisites[i][0]].add(prerequisites[i][1]);
        }

        Queue<Integer> queue = new LinkedList<>();

        for (int i = 0; i < numCourses; i++) {
            if (colors[i] == WHITE) {
                dfs(i, queue, colors, adj);
            }
        }

        if (!isPossible) {
            return new int[0];
        }

        int[] result = new int[queue.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = queue.remove();
        }
        return result;
    }

    @SuppressWarnings("WhileLoopReplaceableByForEach")
    void dfs(int current, Queue<Integer> queue, int[] colors, HashSet<Integer>[] adj) {
        if (!isPossible) {
            return;
        }
        colors[current] = GRAY;
        Integer i;

        Iterator<Integer> iterator = adj[current].iterator();
        while (iterator.hasNext()) {
            i = iterator.next();
            if (colors[i] == WHITE) {
                dfs(i, queue, colors, adj);
            } else if (colors[i] == GRAY) {
                isPossible = false;
                return;
            }
        }

        queue.offer(current);
        colors[current] = BLACK;
    }

    public boolean wordBreakDFS(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>(wordDict);
        Queue<Integer> queue = new LinkedList<>();
        boolean[] visited = new boolean[s.length()];
        queue.add(0);

        while (!queue.isEmpty()) {
            int start = queue.remove();
            if (!visited[start]) {
                for (int end = start + 1; end <= s.length(); end++) {
                    if (set.contains(s.substring(start, end))) {
                        queue.add(end);
                        if (end == s.length()) {
                            return true;
                        }
                    }
                }
                visited[start] = true;
            }
        }
        return false;
    }

    public int numDecodings(String s) {
        char[] c = s.toCharArray();
        if (c.length == 0) {
            return 0;
        }
        return dfs(c, 0);
    }

    private int dfs(char[] c, int index) {
        if (index == c.length) {
            return 1;
        }

        if (c[index] == '0') {
            return 0;
        }

        if (index + 1 == c.length) {
            return dfs(c, index + 1);
        }
        int val = (c[index] - '0') * 10 + (c[index + 1] - '0');
        if (val >= 10 && val <= 26) {
            if (val % 10 == 0) {
                return dfs(c, index + 2);
            }
            return dfs(c, index + 1) + dfs(c, index + 2);
        } else {
            return dfs(c, index + 1);
        }
    }

    class StreamMedian {
        int count = 0;
        int size = 8;
        int[] arr = new int[size];

        public void addNum(int num) {
            if (count == 0) {
                arr[0] = num;
                count++;
                return;
            }
            int numIndex = findIndex(num);
            int[] temp = new int[count++];
            System.arraycopy(arr, 0, temp, 0, count - 1);
            arr[numIndex] = num;
            System.arraycopy(temp, numIndex, arr, numIndex + 1, temp.length - numIndex);
        }

        private int findIndex(int num) {
            int start = 0, end = count - 1;
            while (start <= end) {
                int mid = (start + end) / 2;
                if (num < arr[mid]) {
                    end = mid - 1;
                } else if (num > arr[mid]) {
                    start = mid + 1;
                } else {
                    return mid;
                }
            }

            return start;
        }

        public double findMedian() {
            int mid = count / 2;
            if (count % 2 == 0) {
                return ((double) arr[mid] + arr[mid - 1]) / 2;
            } else {
                return arr[mid];
            }
        }
    }

    public boolean isValidSudoku(char[][] board) {
        for (int i = 0; i < 9; i++) {
            Set<Character> cols = new HashSet<>();
            Set<Character> rows = new HashSet<>();
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.' && !rows.add(board[i][j])) {
                    return false;
                }

                if (board[j][i] != '.' && !cols.add(board[j][i])) {
                    return false;
                }
            }
        }

        for (int r = 0; r < 9; r += 3) {
            for (int c = 0; c < 9; c += 3) {
                Set<Character> box = new HashSet<>();
                for (int i = r; i < r + 3; i++) {
                    for (int j = c; j < c + 3; j++) {
                        if (board[i][j] != '.' && !box.add(board[i][j])) {
                            return false;
                        }
                    }
                }
            }

        }

        return true;

    }

    public boolean isHappy(int n) {
        Set<Integer> set = new HashSet<>();

        int num = n;
        int sum = 0;
        while (num > 0) {
            int mod = num % 10;
            sum += (mod * mod);
            num = num / 10;
            if (num == 0) {
                if (sum == 1) {
                    return true;
                }

                if (set.contains(sum)) {
                    return false;
                } else {
                    set.add(sum);
                    num = sum;
                    sum = 0;
                }
            }
        }
        return false;
    }

    public List<List<String>> groupAnagrams(String[] strs) {

        List<List<String>> result = new ArrayList<>();
        List<Map<Character, Integer>> mapList = new ArrayList<>();
        for (String s : strs) {
            Map<Character, Integer> map = new HashMap<>();
            for (char c : s.toCharArray()) {
                map.put(c, map.getOrDefault(c, 0) + 1);
            }
            mapList.add(map);
        }

        Map<Map<Character, Integer>, Integer> reverseMap = new HashMap<>();
        for (int i = 0; i < mapList.size(); i++) {
            Map<Character, Integer> m = mapList.get(i);
            if (reverseMap.containsKey(m)) {
                int index = reverseMap.get(m);
                result.get(index).add(strs[i]);
            } else {
                List<String> list = new ArrayList<>();
                list.add(strs[i]);
                result.add(list);
                reverseMap.put(m, result.size() - 1);
            }
        }

        return result;
    }

    public int search(int[] nums, int target) {
        if (nums.length == 0) {
            return -1;
        }

        if (nums.length < 2) {
            return nums[0] == target ? 0 : -1;
        }

        int pivot = findPivot(nums);

        if (target > nums[nums.length - 1]) {
            return search(nums, target, 0, pivot - 1);
        } else {
            return search(nums, target, pivot, nums.length - 1);
        }
    }

    int search(int[] arr, int num, int start, int end) {
        if (start > end) {
            return -1;
        }

        int mid = (start + end) / 2;
        if (num < arr[mid]) {
            return search(arr, num, start, mid - 1);
        } else if (num > arr[mid]) {
            return search(arr, num, mid + 1, end);
        } else {
            return mid;
        }

    }

    int findPivot(int[] arr) {
        int start = 0, end = arr.length - 1;
        if (arr[start] < arr[end]) {
            return 0;
        }

        while (start <= end) {
            int pivot = (start + end) / 2;
            if (arr[pivot] > arr[pivot + 1]) {
                return pivot + 1;
            } else {
                if (arr[pivot] < arr[start]) {
                    end = pivot - 1;
                } else {
                    start = pivot + 1;
                }
            }
        }
        return 0;

    }

    public int maxArea(int[] height) {
        if (height.length < 2) {
            return 0;
        }

        int start = 0;
        int end = height.length - 1;
        int area = 0;
        while (start < end) {
            area = Math.max(area, Math.min(height[start], height[end]) * (end - start));
            if (height[start] <= height[end]) {
                start++;
            } else {
                end--;
            }
        }

        return area;
    }

    public String minWindow1(String s, String t) {
        if (s == null || s.isEmpty() || t == null || t.isEmpty()) {
            return "";
        }

        String result = "";
        Map<Character, Integer> mapT = new HashMap<>();
        for (char c : t.toCharArray()) {
            mapT.put(c, mapT.getOrDefault(c, 0) + 1);
        }

        if (mapT.size() > s.length()) {
            return "";
        }

        int start = 0, end = 0, index = 0;
        int mapSCount = 0;
        Map<Character, Integer> mapS = new HashMap<>();

        while (start < s.length()) {
            while (!mapT.containsKey(s.charAt(start))) {
                start++;
                end++;
            }

            while (mapSCount != t.length() && end < s.length()) {
                char c = s.charAt(end);
                if (mapT.containsKey(c)) {
                    mapS.put(c, mapS.getOrDefault(c, 0) + 1);
                    if (mapS.get(c) <= mapT.get(c)) {
                        mapSCount++;
                    }
                }
                end++;
            }

            if (mapSCount == t.length() && (result.isEmpty() || result.length() > (end - start))) {
                result = s.substring(start, end);
                mapSCount = 0;
                mapS.clear();
                System.out.println(result);
            }
            start = end;
        }

        return result;
    }

    public boolean exist(char[][] board, String word) {

        if (word.length() < 1) {
            return true;
        }

        if (board.length == 0 || board[0].length == 0) {
            return false;
        }

        int rows = board.length;
        int cols = board[0].length;

        if (rows * cols < word.length()) {
            return false;
        }

        boolean[][] visited = new boolean[board.length][board[0].length];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (hasNextChar(i, j, board, visited, word, 0)) {
                    return true;
                }
            }
        }

        return false;
    }

    boolean isValid(int r, int c, int rows, int cols, char[][] board, boolean[][] visited, String word, int index) {
        return r >= 0 && r < rows &&
            c >= 0 && c < cols &&
            !visited[r][c] && board[r][c] == word.charAt(index);

    }

    boolean hasNextChar(int r, int c, char[][] board, boolean[][] visited, String word, int index) {
        if (index == word.length()) {
            return true;
        }

        int rows = board.length, cols = board[0].length;

        if (isValid(r, c, rows, cols, board, visited, word, index)) {
            visited[r][c] = true;

            boolean hasNextChar = hasNextChar(r + 1, c, board, visited, word, index + 1) ||
                hasNextChar(r - 1, c, board, visited, word, index + 1) ||
                hasNextChar(r, c + 1, board, visited, word, index + 1) ||
                hasNextChar(r, c - 1, board, visited, word, index + 1);
            visited[r][c] = false;
            return hasNextChar;
        } else {
            return false;
        }
    }

    class RandomizedSet {

        /**
         * Initialize your data structure here.
         */
        public RandomizedSet() {

        }

        /**
         * Inserts a value to the set. Returns true if the set did not already contain the specified element.
         */
        public boolean insert(int val) {
            return false;
        }

        /**
         * Removes a value from the set. Returns true if the set contained the specified element.
         */
        public boolean remove(int val) {
            return false;
        }

        /**
         * Get a random element from the set.
         */
        public int getRandom() {
            return 0;
        }
    }

    /**
     * Your RandomizedSet object will be instantiated and called as such:
     * RandomizedSet obj = new RandomizedSet();
     * boolean param_1 = obj.insert(val);
     * boolean param_2 = obj.remove(val);
     * int param_3 = obj.getRandom();
     */


    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        return levelOrder(root);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data == null || data.isEmpty()) {
            return null;
        }

        String[] nodeVals = data.split(",");
        if (nodeVals.length < 2) {
            return null;
        }

        TreeNode[] nodes = new TreeNode[nodeVals.length];
        for (int i = 1; i < nodes.length; i++) { // 0 is dummy node to make 2k, 2k+1 logic work.
            nodes[i] = getNodeAt(nodeVals, i);
        }

        int i = 1;
        while (2 * i < nodes.length) {
            if (nodes[i] != null) {
                nodes[i].left = nodes[2 * i];
            }
            i++;
        }

        i = 1;
        while (2 * i + 1 < nodes.length) {
            if (nodes[i] != null) {
                nodes[i].right = nodes[2 * i + 1];
            }
            i++;
        }

        return nodes[1];
    }

    TreeNode getNodeAt(String[] nodeVals, int i) {
        String nodeV = nodeVals[i];
        if (nodeV == null || nodeV.isEmpty()) {
            return null;
        } else {
            return new TreeNode(Integer.valueOf(nodeV));
        }
    }

    String levelOrder(TreeNode node) {
        if (node == null) {
            return "";
        }
        StringBuilder result = new StringBuilder();

        List<TreeNode> levelNodes = new ArrayList<>();
        levelNodes.add(node);
        result.append(",").append(node.val).append(",");

        while (!levelNodes.isEmpty()) {
            levelNodes = levelOrder(levelNodes);
            for (TreeNode n : levelNodes) {
                result.append(n == null ? "" : n.val).append(",");
            }
        }

        return result.substring(0, result.length() - 1);
    }

    List<TreeNode> levelOrder(List<TreeNode> list) {
        List<TreeNode> result = new ArrayList<>();
        boolean hasOneElement = false;

        for (TreeNode n : list) {
            if (n == null) {
                result.add(null);
                result.add(null);
            } else {
                result.add(n.left);
                result.add(n.right);
                hasOneElement = hasOneElement || n.left != null || n.right != null;
            }
        }

        return hasOneElement ? result : new ArrayList<>();

    }

    public int lengthOfLongestSubstring(String s) {

        StringBuilder sb = new StringBuilder();
        Set<Character> set = new HashSet<>();

        int maxLength = 0;
        int i = 0;

        while (i < s.length()) {
            char c = s.charAt(i);
            if (set.add(c)) {
                sb.append(c);
                i++;
            } else {
                maxLength = Math.max(maxLength, sb.length());
                i = i - sb.length() + 1;
                sb = new StringBuilder();
                set.clear();
            }
        }

        maxLength = Math.max(maxLength, sb.length());

        return maxLength;
    }


    public List<List<Integer>> threeSum(int[] nums) {

        List<List<Integer>> list = new ArrayList<>();

        if (nums == null || nums.length < 3) {
            return list;
        }

        Arrays.sort(nums);

        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }

            int j = i + 1, k = nums.length - 1;
            int target = -nums[i];
            while (j < k) {
                if (nums[j] + nums[k] == target) {
                    list.add(Arrays.asList(nums[i], nums[j], nums[k]));
                    j++;
                    k--;
                    while (j < k && nums[j] == nums[j - 1]) {
                        j++;
                    }
                    while (j < k && nums[k] == nums[k + 1]) {
                        k--;
                    }

                } else if (nums[j] + nums[k] > target) {
                    k--;
                } else {
                    j++;
                }
            }
        }

        return list;

    }

    public int trap(int[] height) {

        if (height.length < 2) {
            return 0;
        }

        int[] pool = new int[height.length];
        int maxHeight = height[0];
        pool[0] = 0;

        for (int i = 1; i < height.length - 1; i++) {
            if (height[i] < maxHeight) {
                pool[i] = maxHeight - height[i];
            } else {
                pool[i] = 0;
                maxHeight = height[i];
            }
        }

        maxHeight = height[height.length - 1];
        for (int i = height.length - 2; i >= 0; i--) {
            if (height[i] < maxHeight) {
                pool[i] = Math.min(pool[i], maxHeight - height[i]);
            } else {
                maxHeight = height[i];
                pool[i] = 0;
            }
        }

        int ans = 0;
        for (int aPool : pool) {
            ans += aPool;
        }

        return ans;
    }

    public double findMedianSortedArrays1(int[] nums1, int[] nums2) {
        int[] arr = mergeArrays(nums1, nums2);
        int mid = arr.length / 2;
        if (arr.length % 2 == 0) {
            return (double) (arr[mid] + arr[mid - 1]) / 2;
        } else {
            return (double) (arr[mid]);
        }
    }

    public int[] mergeArrays(int[] nums1, int[] nums2) {
        int[] arr = new int[nums1.length + nums2.length];

        int i = 0, j = 0, index = 0;
        while (i < nums1.length && j < nums2.length) {
            if (nums1[i] <= nums2[j]) {
                arr[index++] = nums1[i++];
            } else {
                arr[index++] = nums2[j++];
            }
        }

        while (i < nums1.length) {
            arr[index++] = nums1[i++];
        }

        while (j < nums2.length) {
            arr[index++] = nums2[j++];
        }

        return arr;
    }

    public String largestCharacter(String s) {

        char max = '0';

        Set<Character> set = new HashSet<>();

        for (char c : s.toCharArray()) {
            set.add(c);
        }

        for (Character c : set) {
            char oC = getOtherCase(c);
            if (set.contains(oC)) {
                max = (char) Math.max(max, Character.toLowerCase(c));
            }
        }

        return max == '0' ? "No" : max + "";
    }

    char getOtherCase(char c) {
        if (isLowerCase(c)) {
            return (char) (c - 32);
        } else if (isUpperCase(c)) {
            return (char) (c + 32);
        } else {
            return c;
        }
    }

    boolean isLowerCase(char c) {
        return c >= 97 && c < 123;
    }

    boolean isUpperCase(char c) {
        return c >= 65 && c < 91;
    }

    public int shortestDistance(String[] words, String word1, String word2) {

        Map<String, Integer> map = new HashMap<>();
        Map<String, String> wordsMap = new HashMap<>();
        wordsMap.put(word1, word2);
        wordsMap.put(word2, word1);

        int distance = Integer.MAX_VALUE;

        for (int i = 0; i < words.length; i++) {
            if (word1.equals(words[i]) || word2.equals(words[i])) {
                map.put(words[i], i);
                if (map.containsKey(wordsMap.get(words[i]))) {
                    distance = Math.min(distance, Math.abs(i - map.get(wordsMap.get(words[i]))));
                }
            }
        }

        return distance;

    }

    public String reverseWords(String s) {
        StringBuilder sb = new StringBuilder();

        String[] arr = s.trim().split(" ");
        for (int i = arr.length - 1; i >= 0; i--) {
            if (sb.length() == 0) {
                sb.append(arr[i]);
            } else if (!arr[i].isEmpty()) {
                sb.append(" ").append(arr[i]);
            }
        }
        return sb.toString();
    }

    public int maxSubArray(int[] nums) {

        if (nums.length < 1) {
            return 0;
        }

        if (nums.length < 2) {
            return nums[0];
        }

        int currentMax = nums[0], max = nums[0];

        for (int i = 1; i < nums.length; i++) {
            currentMax = Math.max(nums[i], currentMax + nums[i]);
            max = Math.max(max, currentMax);
        }

        return max;
    }

    class LRUCache {

        LinkedList<Integer> keyList = new LinkedList<>();
        int capacity, size;
        Map<Integer, Integer> map = new HashMap<>();

        public LRUCache(int capacity) {
            this.capacity = capacity;
        }

        public int get(int key) {
            if (map.containsKey(key)) {
                keyList.remove((Integer) key);
                keyList.addLast(key);
                return map.get(key);
            }
            return -1;
        }

        public void put(int key, int value) {
            if (!map.containsKey(key) && size == capacity) {
                int k = keyList.removeFirst();
                map.remove(k);
            } else if (map.containsKey(key)) {
                keyList.remove(key);
            } else {
                size++;
            }

            map.put(key, value);
            keyList.addLast(key);
        }
    }

    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) {
            return "";
        }

        if (strs.length == 1) {
            return strs[0];
        }

        int len = strs[0].length();

        for (int i = 1; i < strs.length; i++) {
            len = Math.min(len, strs[i].length());
        }

        StringBuilder sb = new StringBuilder();
        int i = 0;
        while (i < len) {
            for (int j = 1; j < strs.length; j++) {
                if (strs[j].charAt(i) != strs[j - 1].charAt(i)) {
                    return sb.toString();
                }
            }
            sb.append(strs[0].charAt(i++));

        }

        return sb.toString();
    }

    long concatenationsSum(int[] a) {

        long sum = 0;

        for (int i = 0; i < a.length; i++) {
            sum += a[i] * getMultiplier(a[i]) + a[i];
            for (int j = i + 1; j < a.length; j++) {
                sum += a[i] * getMultiplier(a[j]) + a[j];
                sum += a[j] * getMultiplier(a[i]) + a[i];
            }
        }

        return sum;
    }

    long getMultiplier(int n) {
        int multiplier = 10;
        while (n / 10 != 0) {
            multiplier *= 10;
            n /= 10;
        }
        return multiplier;

    }


    int[][] meanGroups(int[][] a) {

        Map<Integer, List<Integer>> map = new HashMap<>();

        for (int i = 0; i < a.length; i++) {
            int sum = 0;
            for (int j = 0; j < a[i].length; j++) {
                sum += a[i][j];
            }
            int mean = sum / a[i].length;
            if (map.containsKey(mean)) {
                map.get(mean).add(i);
            } else {
                List<Integer> list = new ArrayList<>();
                list.add(i);
                map.put(mean, list);
            }
        }

        int[][] result = new int[map.size()][];
        int index = 0;
        for (List<Integer> list : map.values()) {
            int[] arr = new int[list.size()];
            for (int i = 0; i < arr.length; i++) {
                arr[i] = list.get(i);
            }
            result[index++] = arr;
        }

        return result;
    }

    public void bubbleSort(int[] arr) {

        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }

    }

    public void quickSort(int[] arr) {
        quickSort(arr, 0, arr.length - 1);
    }

    public void quickSort(int[] arr, int low, int high) {

        if (arr == null || arr.length == 0 || low >= high) {
            return;
        }


        int mid = low + (high - low) / 2;
        int pivot = arr[mid];


        int i = low, j = high;

        while (i <= j) {

            while (arr[i] < pivot) {
                i++;
            }


            while (arr[j] > pivot) {
                j--;
            }

            if (i <= j) {
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
                i++;
                j--;
            }
        }

        if (low < j) {
            quickSort(arr, low, j);
        }
        if (high > i) {
            quickSort(arr, i, high);
        }

    }

    public void mergeSort(int[] arr) {
        if (arr.length <= 1) {
            return;
        }

        int length = arr.length;

        int[] left = new int[length / 2];
        int[] right = new int[length - left.length];
        System.arraycopy(arr, 0, left, 0, left.length);
        System.arraycopy(arr, left.length, right, 0, right.length);

        mergeSort(left);
        mergeSort(right);

        merge(left, right, arr);
    }

    public void merge(int[] left, int[] right, int[] result) {
        int iLeft = 0;
        int iRight = 0;
        int iResult = 0;

        while (iLeft < left.length && iRight < right.length) {
            if (left[iLeft] < right[iRight]) {
                result[iResult++] = left[iLeft++];
            } else {
                result[iResult++] = right[iRight++];
            }
        }

        System.arraycopy(left, iLeft, result, iResult, left.length - iLeft);
        System.arraycopy(right, iRight, result, iResult, right.length - iRight);
    }

    public int removeDuplicates(int[] nums) {
        if (nums.length < 3) {
            return nums.length;
        }

        int last = 2, next = 2;

        while (next < nums.length) {
            if (nums[last - 2] != nums[next]) {
                nums[last] = nums[next];
                last++;
            }
            next++;
        }
        return last;
    }

    public void moveZeroes(int[] nums) {
        if (nums == null) {
            return;
        }

        int nextIndex = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) {
                nextIndex = nextIndex >= 0 ? Math.min(nextIndex, i) : i;
            } else if (nextIndex >= 0) {
                nums[nextIndex++] = nums[i];
                nums[i] = 0;
            }
        }

    }

    public int[] findDiagonalOrder(int[][] matrix) {

        int rows = matrix.length;
        int cols = matrix[0].length;

        int[] result = new int[rows * cols];
        int elem = 0, row = 0, col = 0;

        boolean isGoingUp = true;

        while (elem < rows * cols) {
            result[elem++] = matrix[row][col];

            if (isGoingUp) {
                if (row == 0 || col == cols - 1) {
                    isGoingUp = false;
                    if (col != cols - 1) {
                        row = 0;
                        col++;
                    } else {
                        row++;
                        col = cols - 1;
                    }
                } else {
                    row--;
                    col++;
                }
            } else {
                if (col == 0 || row == rows - 1) {
                    isGoingUp = true;
                    if (row != rows - 1) {
                        row++;
                        col = 0;
                    } else {
                        row = rows - 1;
                        col++;
                    }
                } else {
                    row++;
                    col--;
                }
            }
        }

        return result;
    }

    public void solve(char[][] board) {

        if (board == null || board.length == 0) {
            return;
        }

        boolean[][] visited = new boolean[board.length][board[0].length];
        markBorders(board, visited);

        for (int i = 1; i < board.length - 1; i++) {
            for (int j = 1; j < board[0].length - 1; j++) {
                if (board[i][j] == 'O' && !visited[i][j]) {
                    board[i][j] = 'X';
                }
            }
        }

    }

    public void markBorders(char[][] board, boolean[][] visited) {
        // top border
        for (int i = 0; i < board[0].length; i++) {
            dfs(board, 0, i, visited);
        }

        // bottom border
        for (int i = 0; i < board[0].length; i++) {
            dfs(board, board.length - 1, i, visited);
        }

        // left border
        for (int i = 1; i < board.length - 1; i++) {
            dfs(board, i, 0, visited);
        }

        // right border
        for (int i = 1; i < board.length - 1; i++) {
            dfs(board, i, board[0].length - 1, visited);
        }
    }

    public boolean isO(char[][] board, int r, int c, boolean[][] visited) {
        return r >= 0 && r < board.length && c >= 0 && c < board[0].length && board[r][c] == 'O' && !visited[r][c];
    }

    public void dfs(char[][] board, int r, int c, boolean[][] visited) {

        if (isO(board, r, c, visited)) {
            visited[r][c] = true;
            dfs(board, r + 1, c, visited);
            dfs(board, r - 1, c, visited);
            dfs(board, r, c + 1, visited);
            dfs(board, r, c - 1, visited);
        }

    }

    public boolean isPalindrome(int x) {

        List<Integer> list = new ArrayList<>();
        int divider = 10;

        int sign = x < 0 ? -1 : 1;
        x = Math.abs(x);

        while (x != 0) {
            list.add(x % divider);
            x /= 10;
        }

        Integer[] arr = list.toArray(new Integer[0]);
        if (arr.length > 0) {
            arr[arr.length - 1] *= sign;
        }

        for (int i = 0, j = arr.length - 1; i <= j; i++, j--) {
            if (!arr[i].equals(arr[j])) {
                return false;
            }
        }

        return true;
    }

    public boolean isPalindrome(String s) {

        List<Character> chars = new ArrayList<>();
        s = s.toLowerCase();
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (Character.isLetterOrDigit(ch)) {
                chars.add(ch);
            }
        }

        Character[] arr = chars.toArray(new Character[0]);

        for (int i = 0, j = arr.length - 1; i <= j; i++, j--) {
            if (arr[i] != arr[j]) {
                return false;
            }
        }

        return true;
    }

    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }

        ListNode mid = head, fast = head;
        while (fast != null && fast.next != null) {
            mid = mid.next;
            fast = fast.next.next;
        }

        ListNode reverse = reverseList(mid);
        printLinkedList(reverse);
        ListNode curr = head;

        while (reverse != null) {
            if (curr.val != reverse.val) {
                return false;
            }
            curr = curr.next;
            reverse = reverse.next;
        }

        return true;
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || head.next == null || k < 2) {
            return head;
        }

        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode p = dummy;
        int size = 1;
        while (head != null) {
            if (size % k == 0) {
                p = reverse(p, head.next);
                head = p.next;
            } else head = head.next;
            size++;
        }
        return dummy.next;
    }

    private ListNode reverse(ListNode start, ListNode end) {
        ListNode slow = end;
        ListNode fast = start.next;
        while (fast != end) {
            ListNode temp = fast.next;
            fast.next = slow;
            slow = fast;
            fast = temp;
        }
        ListNode res = start.next;
        start.next = slow;
        return res;
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {

        ListNode ref = new ListNode(-1);

        ListNode n1 = l1, n2 = l2, curr = ref;

        while (n1 != null && n2 != null) {
            if (n1.val < n2.val) {
                curr.next = n1;
                n1 = n1.next;
            } else {
                curr.next = n2;
                n2 = n2.next;
            }
            curr = curr.next;
        }

        curr.next = n1 == null ? n2 : n1;

        return ref.next;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {

        ListNode slow = head, fast = head;

        while (n-- > 0 && fast != null) {
            fast = fast.next;
        }

        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }

        slow.next = slow.next.next;
        return head;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) {
            return null;
        }

        if (l1 == null) {
            return l2;
        }

        if (l2 == null) {
            return l1;
        }

        ListNode newHead = new ListNode(0);
        ListNode l = l1, r = l2, curr = newHead;
        int carry = 0;

        while (l != null || r != null) {
            carry += l == null ? 0 : l.val;
            carry += r == null ? 0 : r.val;

            curr.next = new ListNode(carry % 10);
            carry /= 10;
            curr = curr.next;

            l = l == null ? null : l.next;
            r = r == null ? null : r.next;
        }

        return newHead.next;
    }

    public void printLinkedList(ListNode head) {
        System.out.print("\n");
        printList(head);
        System.out.print("\n");
    }

    private void printList(ListNode head) {
        if (head == null) {
            return;
        }

        System.out.print(head.val + "  ");
        printList(head.next);
    }

    int i = 0;

    public ListNode reverseList(ListNode head) {

        if (head == null || head.next == null) {
            return head;
        }

        ListNode node = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return node;
    }

    static class ListNode {
        int val;
        ListNode next;

        ListNode(int val) {
            this.val = val;
        }

        @Override public String toString() {
            return "ListNode{" +
                "val=" + val +
                '}';
        }
    }

    public int calculate(String s) {

        Stack<Integer> stack = new Stack<>();

        int operand = 0;
        int result = 0;
        int sign = 1; // 1 positive, -1 negative

        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (Character.isDigit(ch)) {
                operand = 10 * operand + (ch - '0');
            } else if (ch == '+') {
                result += sign * operand;
                sign = 1;
                operand = 0;
            } else if (ch == '-') {
                result += sign * operand;
                sign = -1;
                operand = 0;
            } else if (ch == '(') {
                stack.push(result);
                stack.push(sign);

                sign = 1;
                result = 0;
            } else if (ch == ')') {
                result += sign * operand;
                result += stack.pop();
                result += stack.pop();
                operand = 0;
            }
        }
        return result + (sign * operand);
    }

    public boolean isValid(String s) {
        if (s == null || s.isEmpty()) {
            return true;
        }

        if (s.length() % 2 == 1) {
            System.out.println("Early");
            return false;
        }

        List<Character> openP = Arrays.asList('[', '(', '{');
        List<Character> closeP = Arrays.asList(']', ')', '}');


        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {

            char c = s.charAt(i);

            if (closeP.contains(c)) {
                int index = closeP.indexOf(c);
                if (stack.empty() || stack.peek() != openP.get(index)) {
                    return false;
                } else {
                    stack.pop();
                }
            } else {
                stack.push(c);
            }
        }

        return stack.empty();
    }


    static class Interval {
        int start, end;

        Interval(int start, int end) {
            this.start = start;
            this.end = end;
        }

        @Override public String toString() {
            return "[" + start + ", " + end + "]";
        }

        int[] toArr() {
            return new int[]{start, end};
        }
    }

    public int[][] merge(int[][] intervals) {

        List<Interval> list = new ArrayList<>();
        for (int[] interval : intervals) {
            list.add(new Interval(interval[0], interval[1]));
        }

        //noinspection Convert2Lambda
        list.sort(new Comparator<Interval>() {
            @Override
            public int compare(Interval i1, Interval i2) {
                return i1.start - i2.start;
            }
        });

        int i = 0;

        while (i + 1 < list.size()) {
            Interval i1 = list.get(i);
            Interval i2 = list.get(i + 1);

            if (i2.start < i1.end) {
                list.set(i, new Interval(i1.start, i2.end));
                list.remove(i2);
            } else {
                i++;
            }
        }

        int[][] result = new int[list.size()][2];
        for (i = 0; i < list.size(); i++) {
            result[i] = list.get(i).toArr();
            System.out.println(Arrays.toString(intervals[i]));
        }

        return result;

    }

    void rotateArr(int[] nums, int k) {
        if (k > nums.length) {
            k = k % nums.length;
        }

        for (int i = 0; i < nums.length; i++) {
            int newIndex = i + k >= nums.length ? Math.abs(i + k - nums.length) : i + k;
            System.out.println("new Index: " + newIndex);
        }
    }

    public boolean checkPossibility(int[] nums) {
        if (nums == null || nums.length <= 2) {
            return true;
        }
        int count = 0;
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] < nums[i - 1]) {
                if (count > 0) {
                    return false;
                }
                count++;
                if (i >= 2 && nums[i] < nums[i - 2]) {
                    nums[i] = nums[i - 1];
                } else {
                    nums[i - 1] = nums[i];
                }
            }
        }
        return true;
    }


    int[] reverse(int[] A) {
        int i = 0, j = A.length - 1;
        while (i < j) {
            int temp = A[i];
            A[i++] = A[j];
            A[j--] = temp;
        }

        return A;
    }


    int max = 0;

    public int longestUnivaluePath(TreeNode root) {
        return longestUnivaluePath(root, 0);
    }

    int longestUnivaluePath(TreeNode node, int value) {
        if (node == null) {
            return 0;
        }

        int left = longestUnivaluePath(node.left, node.val);
        int right = longestUnivaluePath(node.right, node.val);

        max = Math.max(max, left + right);

        if (node.val == value) {
            return Math.max(left, right) + 1;
        }
        return 0;
    }

    public int minDiffInBST(TreeNode root) {
        if (root == null) {
            return 0;
        }

        return minDiffBetweenNodes(root, root, 0);
    }

    public int minDiffBetweenNodes(TreeNode root, TreeNode node, int diff) {
        if (node == null) {
            return diff;
        }

        int d = minDiffBetweenNodes(root, node.left, minDiffBetweenNodes(node.val, root, diff));
        return minDiffBetweenNodes(root, node.right, minDiffBetweenNodes(node.val, root, d));
    }

    public int minDiffBetweenNodes(int nodeVal, TreeNode node, int diff) {
        if (node == null) {
            return diff;
        }

        int d = Math.abs(nodeVal - node.val);
        if (diff == 0 || (d != 0 && d < diff)) {
            diff = d;
        }

        int dl = minDiffBetweenNodes(nodeVal, node.left, diff);
        return minDiffBetweenNodes(nodeVal, node.right, dl);
    }

    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        String seq1 = getLeafSequence(root1, "");
        String seq2 = getLeafSequence(root2, "");
        return (seq1 == null && seq2 == null) || (seq1 != null && seq1.equals(seq2));
    }

    String getLeafSequence(TreeNode node, String leafSequence) {
        if (node == null) {
            return leafSequence;
        }

        if (node.left == null && node.right == null) {
            leafSequence += node.val;
        }

        String leftSequence = getLeafSequence(node.left, leafSequence);
        return getLeafSequence(node.right, leftSequence);
    }

    TreeNode increasingBST(TreeNode root) {
        return increasingBST(root, null);
    }

    TreeNode increasingBST(TreeNode node, TreeNode left) {

        if (node == null) {
            return left;
        }

        TreeNode newRoot = increasingBST(node.left, node);
        node.left = null;
        node.right = increasingBST(node.right, left);
        return newRoot;
    }


    boolean cousins(TreeNode root, int x, int y) {
        TreeNodeMeta metaX = getLevelAndParent(root, x, 1, null);
        TreeNodeMeta metaY = getLevelAndParent(root, y, 1, null);

        return metaX.level == metaY.level && metaX.parent != metaY.parent;
    }

    private class TreeNodeMeta extends TreeNode {
        TreeNode parent;
        int level;

        TreeNodeMeta(TreeNode node, int level, TreeNode parent) {
            super(node.val);
            left = node.left;
            right = node.right;
            this.parent = parent;
            this.level = level;
        }
    }

    TreeNodeMeta getLevelAndParent(TreeNode node, int val, int level, TreeNode parent) {
        if (node == null) {
            return null;
        }

        if (node.val == val) {
            return new TreeNodeMeta(node, level, parent);
        }

        TreeNodeMeta nodeP = getLevelAndParent(node.left, val, level + 1, node);
        if (nodeP == null) {
            nodeP = getLevelAndParent(node.right, val, level + 1, node);
        }

        return nodeP;
    }

    TreeNode parent(TreeNode node, int val, TreeNode parent) {
        if (node == null) {
            return null;
        }

        if (node.val == val) {
            return parent;
        }

        TreeNode nodeP = parent(node.left, val, node);
        if (nodeP == null) {
            nodeP = parent(node.right, val, node);
        }

        return nodeP;
    }

    int level(TreeNode node, int val, int level) {
        if (node == null) {
            return 0;
        }

        if (node.val == val) {
            return level;
        }

        int nodeLevel = level(node.left, val, level + 1);
        if (nodeLevel == 0) {
            nodeLevel = level(node.right, val, level + 1);
        }

        return nodeLevel;
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int value) {
            val = value;
        }

        @Override public String toString() {
            return "" + val;
//            return "TreeNode{" +
//                "val=" + val +
//                ", left=" + left +
//                ", right=" + right +
//                '}';
        }
    }
}
