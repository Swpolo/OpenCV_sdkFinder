package paul.opencv_sdkfinder;

import org.junit.Test;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;

import static org.junit.Assert.*;

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
public class ExampleUnitTest {
//    @Test
//    public void addition_isCorrect() throws Exception {
//        assertEquals(4, 2 + 2);
//    }

    /// TODO
//    @Test
//    public void orderPoint_isCorrect() throws Exception {
//        MatOfPoint2f src;
//        MatOfPoint2f dst;
//        orderPointCpp(src.getNativeObjAddr(), dst.getNativeObjAddr());
//    }

    public native boolean orderPointCpp(long matAddrSrc, long matAddrDst);
}