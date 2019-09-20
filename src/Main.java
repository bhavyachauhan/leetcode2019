import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;

@SuppressWarnings({"unused", "WeakerAccess", "ForLoopReplaceableByForEach"})
public class Main {

    public static void main(String[] args) {

        int[][] graph = new int[][]{{1, 0}, {2, 0}, {3, 1}, {3, 2}};
        System.out.println(Arrays.toString(new Main().findOrder(4, graph)));

    }

    static int WHITE = 0;
    static int GRAY = 1;
    static int BLACK = 2;

    boolean isPossible = true;
    int[] result;
    int index = 0;

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        result = new int[numCourses];

        @SuppressWarnings("unchecked")
        HashSet<Integer>[] adj = new HashSet[numCourses];
        int[] colors = new int[numCourses];

        for (int i = 0; i < numCourses; i++) {
            adj[i] = new HashSet<>();
        }

        for (int[] course: prerequisites) {
            adj[course[0]].add(course[1]);
        }

        for (int i = 0; i < numCourses; i++) {
            if (colors[i] == WHITE) {
                dfs(i, colors, adj);
            }
        }

        if (!isPossible) {
            return new int[0];
        }

        return result;
    }

    void dfs(int current, int[] colors, HashSet<Integer>[] adj) {
        if (!isPossible) {
            return;
        }

        colors[current] = GRAY;

        Iterator<Integer> it = adj[current].iterator();
        Integer i;
        while(it.hasNext()) {
            i = it.next();
            if (colors[i] == GRAY) {
                isPossible = false;
                return;
            } else if (colors[i] == WHITE) {
                dfs(i, colors, adj);
            }
        }

        colors[current] = BLACK;
        result[index++] = current;
    }

    private static void exchange(int[] pq, int i, int j) {
        int temp = pq[i];
        pq[i] = pq[j];
        pq[j] = temp;
    }


    public int shipWithinDays(int[] weights, int D) {
        int start = 0, end = weightSum(weights);

        while(start <= end) {
            int mid = (start + end) / 2;
            int daysForCapacity = daysForCapacity(weights, mid);

            if (daysForCapacity == -1 || daysForCapacity > D){
                start = mid + 1;
            } else if (daysForCapacity < D) {
                end = mid - 1;
            } else {
                while(daysForCapacity == D) {
                    daysForCapacity = daysForCapacity(weights, --mid);
                }
                return mid + 1;
            }
        }

        return -1;
    }

    int weightSum(int[] weights) {
        int total = 0;
        for (int i: weights) {
            total += i;
        }
        return total;
    }

    int daysForCapacity(int[] weights, int capacity) {

        int days = 1;
        int currentWeight = 0;

        for (int i = 0; i< weights.length; i++) {
            if (weights[i] > capacity) {
                return -1;
            }
            currentWeight += weights[i];
            if (currentWeight > capacity) {
                days++;
                currentWeight = weights[i];
            }
        }

        return days;

    }

    public int[] maxDepthAfterSplit(String seq) {

        int[] result = new int[seq.length()];

        char[] arr = seq.toCharArray();

        int counter = 0;

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == '(') {
                result[i] = counter % 2;
                counter++;
            } else if (arr[i] == ')') {
                counter--;
                result[i] = counter % 2;
            }
        }

        return result;
    }

    public int peakIndexInMountainArray(int[] A) {
        if (A.length < 3) {
            return -1;
        }

        int start = 0, end = A.length - 1;

        while (start <= end) {
            int mid = (start + end) / 2;
            if (A[mid] > A[mid + 1] && A[mid] > A[mid - 1]) {
                return mid;
            } else if (A[mid] < A[mid + 1]) {
                start = mid;
            } else if (A[mid] < A[mid - 1]) {
                end = mid;
            }
        }
        return -1;
    }

    public char nextGreatestLetter(char[] letters, char target) {

        int start = 0, end = letters.length - 1;

        while (start < end) {

            int mid = start + ((end - start) / 2);

            if (letters[mid] > target) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }

        return letters[end] <= target ? letters[0] : letters[end];
    }

    public int search(int[] nums, int target) {

        int start = 0, end = nums.length - 1;

        while (start <= end) {

            int mid = (start + end) / 2;

            if (nums[mid] == target) {
                return mid;
            }

            if (nums[mid] > target) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }

        return -1;
    }

    public int findRadius(int[] houses, int[] heaters) {

        if (houses.length == 0) return 0;
        if (heaters.length == 0) return Integer.MAX_VALUE;

        Arrays.sort(houses);
        Arrays.sort(heaters);

        int radius = 0;

        for (int i : houses) {
            int distance = binarySearchClosestHeater(heaters, i);
            if (distance > 0 && distance >= radius) {
                radius = distance;
            }
        }

        return radius;

    }

    // returns distance from closest heater
    int binarySearchClosestHeater(int[] heaters, int house) {

        int start = 0, end = heaters.length - 1;

        while (start + 1 < end) {
            int mid = start + ((end - start) / 2);
            if (heaters[mid] == house) {
                return 0;
            }

            if (heaters[mid] > house) {
                end = mid;
            } else {
                start = mid;
            }
        }

        int left = Math.abs(heaters[start] - house);
        int right = Math.abs(heaters[end] - house);

        return Math.min(left, right);
    }

    public int arrangeCoins(int n) {
        int i = n, j = 1, result = 0;
        while (true) {
            i = i - j;
            j++;
            if (i >= 0) {
                result++;
            } else {
                return result;
            }
        }
    }

    public boolean isPerfectSquare(int x) {
        int start = 0, end = x;

        while (start < end) {
            int mid = ((start + end) / 2) + 1;

            if (x / mid == mid && x % mid == 0) {
                return true;
            }

            if (mid > x / mid) {
                end = mid - 1;
            } else {
                start = mid;
            }
        }
        return false;

    }

    public int sqrt(int x) {
        int start = 0, end = x;

        while (start < end) {
            int mid = ((start + end) / 2) + 1;

            if (mid > x / mid) {
                end = mid - 1;
            } else {
                start = mid;
            }
        }

        return start;
    }

    public int[] intersection(int[] nums1, int[] nums2) {
        int length1 = nums1.length, length2 = nums2.length;
        int[] result = new int[Math.min(length1, length2)];

        Arrays.sort(nums1);
        Arrays.sort(nums2);

        int i = 0, j = 0, k = 0;

        while (i < length1 && j < length2) {
            if (nums1[i] < nums2[j]) {
                i++;
            } else if (nums1[i] > nums2[j]) {
                j++;
            } else {
                if (k == 0 || result[k - 1] != nums1[i]) {
                    result[k++] = nums1[i];
                }
                i++;
                j++;
            }
        }

        return result;

    }

    public int firstBadVersion(int n) {
        return firstBadVersion(0, n);
    }

    int firstBadVersion(int start, int end) {
        int firstBad = -1;

        if (start == end) {
            return isBadVersion(start) ? start : -1;
        }

        int mid = start + ((end - start) / 2);
        if (isBadVersion(mid)) {
            return firstBadVersion(start, mid);
        } else {
            return firstBadVersion(mid + 1, end);
        }

    }

    @SuppressWarnings("FieldCanBeLocal")
    int badVersion = 4;
    boolean isBadVersion(int version) {
        return version >= badVersion;
    }

    public int[] twoSum(int[] nums, int target) {
        int[] index = new int[2];
        int i = 0;
        int j = nums.length - 1;
        while (i < j) {
            int t = nums[i] + nums[j];
            if (t < target)
                i++;
            else if (t == target) {
                index[0] = i + 1;
                index[1] = j + 1;
                return index;
            } else {
                j--;
            }
        }
        return index;
    }

    public int searchInsert(int[] nums, int target) {
        int startIndex = 0;
        int endIndex = nums.length - 1;

        while (startIndex <= endIndex) {
            int mid = (startIndex + endIndex) / 2;
            if (target > nums[mid]) {
                startIndex = mid + 1;
            } else if (target < nums[mid]) {
                endIndex = mid - 1;
            } else {
                return mid;
            }
        }

        return startIndex;

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

    private void reverse(int[] arr) {

        int n = arr.length;
        int t;
        for (int i = 0; i < n / 2; i++) {
            t = arr[i];
            arr[i] = arr[n - i - 1];
            arr[n - i - 1] = t;
        }
    }

    private void invert(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            arr[i] = Math.abs(arr[i] - 1);
        }
    }


}
