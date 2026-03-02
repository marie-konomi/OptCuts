//
//  SigmaBoundEnergy.hpp
//  OptCuts
//
//  Penalty energy for constraining singular values σ1, σ2 of the
//  per-triangle Jacobian (3D rest → 2D UV) individually.
//  E = Σ_tri w_tri * ( max(0, σ1 - σ1_max)^2 + max(0, σ2 - σ2_max)^2 )
//

#ifndef SigmaBoundEnergy_hpp
#define SigmaBoundEnergy_hpp

#include "Energy.hpp"

namespace OptCuts {

    class SigmaBoundEnergy : public Energy
    {
    public:
        SigmaBoundEnergy(double sigma1_max, double sigma2_max,
                        double sigma1_min = 0.0, double sigma2_min = 0.0);

        virtual void getEnergyValPerElem(const TriMesh& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight = false) const;
        virtual void getEnergyValByElemID(const TriMesh& data, int elemI, double& energyVal, bool uniformWeight = false) const;

        virtual void computeGradient(const TriMesh& data, Eigen::VectorXd& gradient, bool uniformWeight = false) const;

        virtual void computeHessian(const TriMesh& data, Eigen::VectorXd* V,
                                    Eigen::VectorXi* I = NULL, Eigen::VectorXi* J = NULL, bool uniformWeight = false) const;
        virtual void computeHessian(const TriMesh& data,
                                    Eigen::MatrixXd& Hessian,
                                    bool uniformWeight = false) const;

        virtual void checkEnergyVal(const TriMesh& data) const;

        /// Compute actual max singular values over all triangles (for logging).
        static void getMaxSingularValues(const TriMesh& data, double& max_sigma1, double& max_sigma2);

        /// Compute 2x2 Jacobian J (rest 2D -> UV) and singular values σ1 ≥ σ2 for one triangle.
        /// Returns false if triangle is degenerate. (Public for use by SigmaSmoothEnergy.)
        static bool computeJacobianAndSingularValues(const TriMesh& data, int triI,
            Eigen::Matrix2d& J, double& sigma1, double& sigma2);

    private:
        double sigma1_max_;
        double sigma2_max_;
        double sigma1_min_;
        double sigma2_min_;
    };
}

#endif /* SigmaBoundEnergy_hpp */
