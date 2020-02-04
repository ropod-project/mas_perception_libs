/*!
 * @copyright 2018 Bonn-Rhein-Sieg University
 *
 * @author Minh Nguyen
 *
 */
#ifndef MAS_PERCEPTION_LIBS_POINT_CLOUD_UTILS_ROS_H
#define MAS_PERCEPTION_LIBS_POINT_CLOUD_UTILS_ROS_H

#include <string>
#include <opencv/cv.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <mas_perception_msgs/PlaneList.h>
#include <mas_perception_libs/color.h>
#include <mas_perception_libs/PlaneFittingConfig.h>
#include <mas_perception_libs/bounding_box_2d.h>
#include <mas_perception_libs/point_cloud_utils.h>
#include <mas_perception_libs/sac_plane_segmenter.h>

namespace mas_perception_libs
{
/*!
 * @brief extract a CV image from a sensor_msgs/PointCloud2 object
 */
cv::Mat
cloudMsgToCvImage(const sensor_msgs::PointCloud2 &pCloudMsg);

/*!
 * @brief extract a CV matrix containing (x, y, z) coordinates from a sensor_msgs/PointCloud2 object within a region
 *        defined by a BoundingBox2D object
 * @param pBox: bounding box for cropping cloud, may be adjusted to fit the cloud width and height
 */
cv::Mat
cropCloudMsgToXYZ(const sensor_msgs::PointCloud2 &pCloudMsg, BoundingBox2D &pBox);

/*!
 * @brief crops a sensor_msgs/PointCloud2 object object to a region defined by a BoundingBox2D object
 * @param pBox: bounding box for cropping cloud, may be adjusted to fit the cloud width and height
 * @param pCroppedCloudMsg: cropped cloud returned as parameter
 */
void
cropOrganizedCloudMsg(const sensor_msgs::PointCloud2 &pCloudMsg, BoundingBox2D &pBox,
                      sensor_msgs::PointCloud2& pCroppedCloudMsg);

/*!
 * @brief creates a visualization marker for the plane convex hull
 * @param pPlaneMsg: plane message to visualize
 * @param pNamespace: marker namespace
 * @param pColor: marker color
 * @param pThickness: marker thickness in Rviz (meter)
 * @param pId: marker ID
 * @return: visualization marker of the plane's convex hull
 */
visualization_msgs::Marker::Ptr
planeMsgToMarkers(const mas_perception_msgs::Plane &pPlaneMsg, const std::string &pNamespace,
                  Color pColor = Color(Color::TEAL), float pThickness = 0.005, int pId = 1);

/*!
 * @brief converts PlaneModel object to a plane message
 */
mas_perception_msgs::Plane::Ptr
planeModelToMsg(const PlaneModel &pModel);

// sensor_msgs::PointCloud2::ConstPtr
float
getDominantOrientation(const sensor_msgs::PointCloud2::ConstPtr &pCloudPtr,
                       const std::vector<double> &referenceNormal,
                       double angleFilterTolerance);


/*!
 * @brief combines plane segmentation and cloud filtering and allow configuration using dynamic reconfigure
 */
class PlaneSegmenterROS
{
public:
    PlaneSegmenterROS() = default;

    /*!
     * @brief use dynamic reconfigure params in PlaneFittingConfig to configure parameters of CloudFilter and
     * SacPlaneSegmenter objects
     */
    virtual void
    setParams(const PlaneFittingConfig &pConfig);

    /*!
     * @brief only run cloud filtering
     * @param pCloudPtr: (in) pointer to cloud to be filtered
     * @return: point to filtered cloud
     */
    virtual sensor_msgs::PointCloud2::Ptr
    filterCloud(const sensor_msgs::PointCloud2::ConstPtr &pCloudPtr);

    /*!
     * @brief perform cloud filtering and fitting multiple planes
     *        TODO(minhnh): fitting multiple planes is not implemented
     * @param pCloudPtr: (in) pointer to cloud to be processed
     * @param pFilteredCloudMsg: (out) pointer to filtered cloud
     * @return pointer to a list of detected planes
     */
    virtual mas_perception_msgs::PlaneList::Ptr
    findPlanes(const sensor_msgs::PointCloud2::ConstPtr &pCloudPtr, sensor_msgs::PointCloud2::Ptr &pFilteredCloudMsg);


protected:
    CloudFilter mCloudFilter;
    SacPlaneSegmenter mPlaneSegmenter;
};

}   // namespace mas_perception_libs

#endif  // MAS_PERCEPTION_LIBS_POINT_CLOUD_UTILS_ROS_H
