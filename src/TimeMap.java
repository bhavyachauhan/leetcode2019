import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

public class TimeMap {

    Map<String, Map<Integer, String>> map = new HashMap<>();

    public TimeMap() {

    }

    public void set(String key, String value, int timestamp) {
        final Map<Integer, String> values;
        if (map.containsKey(key)) {
            values = map.get(key);
        } else {
            values = new HashMap<>();
        }
        values.put(timestamp, value);
        map.put(key, values);
    }

    public String get(String key, int timestamp) {
        if (map.containsKey(key)) {
            final Map<Integer, String> values = map.get(key);
            if (values.containsKey(timestamp)) {
                return values.get(timestamp);
            } else if (values.isEmpty()) {
                return "";
            } else {
                return values.get(getClosestTimeStamp(values.keySet(), timestamp));
            }
        } else {
            return "";
        }
    }

    private int getClosestTimeStamp(Set<Integer> set, int timestamp) {
        final int[] timestamps = new int[set.size()];
        final Iterator<Integer> iterator = set.iterator();

        for (int i = 0; i < timestamps.length; i++) {
            timestamps[i] = iterator.next();
        }
        Arrays.sort(timestamps);

        int start = 0, end = timestamps.length - 1;

        while(start <= end) {
            int mid = (start + end) / 2;

            if (timestamps[mid] == timestamp) {
                return timestamps[mid];
            }

            if (timestamps[mid] > timestamp) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }

        return timestamps[start];
    }

}
